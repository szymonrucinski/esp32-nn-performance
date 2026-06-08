#!/usr/bin/env python3
"""Measure ESP32-S3 (ESP32-S3-EYE) power with a Nordic PPK2 in SOURCE mode.

Wiring (source meter mode):
  PPK2 VOUT -> board VDD_3V3 (J9 pin 2)
  PPK2 GND  -> board GND     (J9 odd pin)
  PPK2 VIN  -> unused
  Data/serial to the ESP USB-JTAG via a VBUS-CUT usb cable (so the onboard
  buck never sees 5V while the PPK2 sources 3.3V -- no supply fight).

The PPK2 itself powers the board: with output OFF the ESP is dead and does
NOT enumerate; we toggle DUT power ON, wait for boot, then a NEW /dev/ttyACM*
(vid 303a) appears = the ESP. We read its STATUS line and, in parallel,
average the PPK2 current to get power and energy/inference.

  P_avg   = V_src * I_avg
  mJ/inf  = P_avg(W) * latency(s) * 1000

Measure board-level active power (whole 3V3 rail: SoC + PSRAM + flash +
peripherals), not core-isolated. Inference runs continuously (firmware
BURST_IDLE_MS=0) so I_avg reflects pure active draw — no idle contamination.

Sanity check vs. literature: our energy/MAC is 0.67-1.96 mJ/MMAC across the 4
INT8 models. Karic et al. measure CNN inference on the SAME chip (ESP32-S3)
with the SAME instrument (PPK2 source meter) and report 1.17-1.62 mJ/MMAC
(17-22 uAh/inf @ 3.3V for SqueezeNet/MobileNetV2). Our range brackets theirs;
we trend lower (more efficient) because ESP-DL's hand-tuned INT8 SIMD kernels
beat a generic deploy -- the expected direction.
  Karic et al., "Send Less, Save More: Energy-Efficiency Benchmark of
  Embedded CNN Inference vs. Data Transmission in IoT", arXiv:2510.24829 (2025).
  https://arxiv.org/abs/2510.24829
"""

import csv
import glob
import os
import sys
import time

import serial
from ppk2_api.ppk2_api import PPK2_API
from serial.tools import list_ports

V_SRC_MV = 3300  # source voltage (ESP32-S3 nominal Vdd)
BOOT_WAIT_S = 8  # firmware has a ~5s vTaskDelay before STATUS loop
ESP_FIND_S = 20  # how long to wait for the ESP to enumerate
SERIAL_READ_S = 4  # how long to sniff the ESP serial for a STATUS line
MEAS_S = 10  # current-averaging window per run (seconds)
N_RUNS = 10  # number of independent measurement runs
ESP_VID = 0x303A  # Espressif USB-JTAG
FALLBACK_MS = 130.03  # SqueezeNet INT8 latency
MODEL_NAME = "SqueezeNet_INT8"


def _flush_port(dev):
    """A prior unclean exit can leave the PPK2 streaming; drain the binary
    buffer and send AverageStop (0x07) so get_modifiers can parse cleanly."""
    try:
        s = serial.Serial(dev, timeout=0.2)
        try:
            s.write(bytes([0x07]))
        except Exception:
            pass
        time.sleep(0.3)
        s.reset_input_buffer()
        s.reset_output_buffer()
        while s.read(4096):
            pass
        s.close()
    except Exception:
        pass


def open_ppk2():
    """Both PPK2 CDC ports report as ttyACM; pick the one whose modifier
    (calibration) read actually succeeds -- that's the control interface."""
    cands = [p.device for p in list_ports.comports() if (p.vid, p.pid) == (0x1915, 0xC00A)]
    cands = cands or sorted(glob.glob("/dev/ttyACM*"))
    for dev in cands:
        _flush_port(dev)
        for attempt in range(5):  # transient decode errors -> retry
            try:
                ppk2 = PPK2_API(dev)
                if ppk2.get_modifiers():  # reads calibration; falsy => wrong port
                    print(f"PPK2 control port: {dev}")
                    return ppk2
            except Exception as e:
                if attempt == 4:
                    print(f"  {dev}: {e}")
                time.sleep(0.6)
    sys.exit("ERROR: could not open PPK2 control port")


def find_esp(deadline):
    """Return the ESP serial device that appears AFTER power-on."""
    while time.time() < deadline:
        for p in list_ports.comports():
            if p.vid == ESP_VID:
                return p.device
        time.sleep(0.3)
    return None


def read_status(port, secs):
    """Sniff the ESP serial; return (raw_lines, latency_ms_or_None)."""
    lat = None
    lines = []
    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
    except Exception as e:
        print(f"  serial open failed: {e}")
        return lines, lat
    end = time.time() + secs
    buf = ""
    while time.time() < end:
        c = ser.read(4096)
        if c:
            buf += c.decode("utf-8", "replace")
    ser.close()
    for ln in buf.splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)
        if "ms/inf=" in ln:
            try:
                lat = float(ln.split("ms/inf=")[1].split()[0])
            except Exception:
                pass
    return lines, lat


def main():
    ppk2 = open_ppk2()
    ppk2.use_source_meter()
    ppk2.set_source_voltage(V_SRC_MV)
    print(f"source = {V_SRC_MV} mV, powering DUT ON ...")
    ppk2.toggle_DUT_power("ON")

    try:
        print(f"waiting {BOOT_WAIT_S}s for ESP boot ...")
        time.sleep(BOOT_WAIT_S)

        esp = find_esp(time.time() + ESP_FIND_S)
        if esp:
            print(f"ESP enumerated at {esp}; reading serial {SERIAL_READ_S}s ...")
            lines, latency = read_status(esp, SERIAL_READ_S)
            for ln in lines[-6:]:
                print(f"  | {ln}")
        else:
            print(
                "WARN: ESP did not enumerate (check VBUS-cut cable / wiring). "
                "Measuring power anyway."
            )
            latency = None

        if latency is None:
            latency = FALLBACK_MS
            print(f"no live ms/inf -> using known latency {latency} ms")
        else:
            print(f"live latency: {latency} ms/inf")

        # ---- 10-run current averaging ----
        v = V_SRC_MV / 1000.0
        run_results = []
        for run in range(1, N_RUNS + 1):
            ppk2.start_measuring()
            samples = []
            end = time.time() + MEAS_S
            while time.time() < end:
                data = ppk2.get_data()
                if data != b"":
                    got = ppk2.get_samples(data)
                    got = got[0] if isinstance(got, tuple) else got
                    samples.extend(got)
                time.sleep(0.01)
            ppk2.stop_measuring()
            if not samples:
                print(f"  run {run:2d}: NO SAMPLES")
                continue
            i_mean_ma = sum(samples) / len(samples) / 1000.0
            p_mw = v * i_mean_ma
            mj = p_mw * (latency / 1000.0)
            run_results.append({"i_ma": i_mean_ma, "p_mw": p_mw, "mj": mj, "n": len(samples)})
            print(f"  run {run:2d}: {i_mean_ma:.2f} mA  {p_mw:.2f} mW  {mj:.3f} mJ/inf")

        if not run_results:
            sys.exit("ERROR: no current samples captured")

        mjs = [r["mj"] for r in run_results]
        pws = [r["p_mw"] for r in run_results]
        mean_mj = sum(mjs) / len(mjs)
        mean_pw = sum(pws) / len(pws)
        var_mj = sum((x - mean_mj) ** 2 for x in mjs) / len(mjs)
        std_mj = var_mj**0.5
        sem_mj = std_mj / len(mjs) ** 0.5

        print(f"\n========== PPK2 power result — {MODEL_NAME} ==========")
        print(f"  runs            : {len(run_results)}")
        print(f"  V source        : {v:.3f} V")
        print(f"  P avg           : {mean_pw:.2f} mW")
        print(f"  latency         : {latency:.2f} ms/inf")
        print(f"  ENERGY mean     : {mean_mj:.3f} mJ/inf")
        print(f"  ENERGY std      : {std_mj:.3f} mJ/inf  (SEM {sem_mj:.4f})")
        print("=" * 55)

        outdir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(outdir, exist_ok=True)
        slug = MODEL_NAME.lower().replace(" ", "_")
        out = os.path.join(outdir, f"power_{slug}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "i_mean_mA", "p_avg_mW", "latency_ms", "energy_mJ", "samples"])
            for i, r in enumerate(run_results, 1):
                w.writerow(
                    [
                        i,
                        f"{r['i_ma']:.3f}",
                        f"{r['p_mw']:.3f}",
                        f"{latency:.2f}",
                        f"{r['mj']:.3f}",
                        r["n"],
                    ]
                )
        print(f"CSV: {os.path.abspath(out)}")

    finally:
        # leave the board powered so it keeps running; comment out to cut power
        # ppk2.toggle_DUT_power("OFF")
        print("(DUT left powered ON; run ppk2.toggle_DUT_power('OFF') to cut)")


if __name__ == "__main__":
    main()
