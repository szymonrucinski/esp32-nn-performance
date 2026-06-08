// EuroSAT classifier + benchmark firmware for ESP32-S3 (Xtensa LX7) via ESP-DL.
// One image is classified to verify the deploy path, then the firmware runs
// BOTH benchmark phases so a single build serves both host tools:
//
//   Phase 1 (latency)  — NUM_WARMUP warmups + NUM_TIMED timed runs, then prints
//       one line:  result=[OK <name> ms/inf=.. MAC/cycle=.. min=.. max=..]
//       parsed by espdl_bench/bench.sh into results/bench_results.csv.
//   Phase 2 (power)    — continuous inference forever, emitting per-inference
//       STATUS lines, sampled by espdl_bench/measure_power_ppk2.py via a PPK2.
//
// Energy (mJ/inf) is NOT estimated on-device; it is measured externally with
// the PPK2 (P_avg x latency). No fixed-power guess lives here.
#include "dl_model_base.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_image_define.hpp"
#include "eurosat_cls_postprocessor.hpp"
#include "eurosat_category_name.hpp"
#include "test_image.h"
#include "model_config.h"   // MODEL_NAME, MODEL_MMAC (written per-model by bench.sh)
#include <cmath>
#include <cstdint>

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstdio>
#include <string>

// Phase 1 (latency) constants.
#define NUM_WARMUP     3         // discarded runs to prime caches/PSRAM
#define NUM_TIMED      10        // timed runs averaged for ms/inf
#define CPU_FREQ_MHZ   240       // Xtensa LX7 clock, for MAC/cycle

// Phase 2 (power) protocol, read by espdl_bench/measure_power_ppk2.py:
//   STATUS run=<r> inf=<k> ms/inf=<t> true=<l> pred=<p>   (one per inference)
//   STATUS run=<r> burst_done                             (end of a burst)
#define BURST_SIZE     10        // inferences per burst
#define BURST_IDLE_MS  0         // no idle — continuous inference for clean power measurement

extern const uint8_t espdl_model_start[] asm("_binary_model_espdl_start");

static const char *TAG = "eurosat";
static char result_buf[256] = "no result yet";
static const char *err_msg = "none";

extern "C" void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(5000));  // let USB-JTAG serial settle after reset

    dl::Model *model = new dl::Model((const char *)espdl_model_start,
                                     fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (!model) { err_msg = "model load failed"; goto idle; }
    model->minimize();

    {
        // [0,1] training normalization: (pixel - 0) / 255. mean/std are in [0,255]
        // units per the ESP-DL API. Wrong norm here = the 24%-accuracy bug.
        dl::image::ImagePreprocessor preprocessor(
            model, {0.0f, 0.0f, 0.0f}, {255.0f, 255.0f, 255.0f}, /*rgb_swap=*/false);

        // top-3, no score floor, softmax on (model outputs raw logits).
        dl::cls::EuroSatClsPostprocessor postprocessor(model, 3, 0.0f, true);

        dl::image::img_t img;
        img.data = (void *)test_image_rgb888;
        img.width = TEST_IMG_W;
        img.height = TEST_IMG_H;
        img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;

        preprocessor.preprocess(img);   // fills + quantizes model input tensor

        // --- DEBUG: inspect quantized input tensor ---
        auto &ins = model->get_inputs();
        dl::TensorBase *in = ins.begin()->second;
        int in_exp = (int)in->exponent;
        ESP_LOGI(TAG, "input '%s' exp=%d dtype=%d size=%d",
                 ins.begin()->first.c_str(), in_exp, (int)in->dtype, in->get_size());
        if (in->dtype == dl::DATA_TYPE_INT16) {
            int16_t *p = (int16_t *)in->data; int16_t mn = p[0], mx = p[0];
            for (int i = 0; i < in->get_size(); i++) { if (p[i] < mn) mn = p[i]; if (p[i] > mx) mx = p[i]; }
            ESP_LOGI(TAG, "  in[0..5]=%d %d %d %d %d %d  min=%d max=%d (deq min=%.4f max=%.4f)",
                     p[0], p[1], p[2], p[3], p[4], p[5], mn, mx,
                     (double)(mn * exp2f(in_exp)), (double)(mx * exp2f(in_exp)));
        } else if (in->dtype == dl::DATA_TYPE_INT8) {
            int8_t *p = (int8_t *)in->data; int8_t mn = p[0], mx = p[0];
            for (int i = 0; i < in->get_size(); i++) { if (p[i] < mn) mn = p[i]; if (p[i] > mx) mx = p[i]; }
            ESP_LOGI(TAG, "  in[0..5]=%d %d %d %d %d %d  min=%d max=%d (deq min=%.4f max=%.4f)",
                     p[0], p[1], p[2], p[3], p[4], p[5], mn, mx,
                     (double)(mn * exp2f(in_exp)), (double)(mx * exp2f(in_exp)));
        }

        model->run();

        // --- DEBUG: raw output logits ---
        auto &outs = model->get_outputs();
        dl::TensorBase *o = outs.begin()->second;
        int o_exp = (int)o->exponent;
        ESP_LOGI(TAG, "output '%s' exp=%d dtype=%d size=%d",
                 outs.begin()->first.c_str(), o_exp, (int)o->dtype, o->get_size());
        for (int i = 0; i < o->get_size() && i < 10; i++) {
            float v = (o->dtype == dl::DATA_TYPE_INT16) ? ((int16_t *)o->data)[i] * exp2f(o_exp)
                    : (o->dtype == dl::DATA_TYPE_INT8)  ? ((int8_t *)o->data)[i] * exp2f(o_exp)
                    : ((float *)o->data)[i];
            ESP_LOGI(TAG, "  logit[%d] %s = %.4f", i, eurosat_cat_names[i], (double)v);
        }

        // Guard: postprocessor indexes eurosat_cat_names[argmax] which only has 10
        // entries. Skip for non-EuroSAT models (e.g. MCUNetV1 with 1000 outputs).
        const char *top1 = "N/A";
        float top1_score = 0.0f;
        if (o->get_size() == 10) {
            auto &results = postprocessor.postprocess();
            if (!results.empty()) {
                top1 = results[0].cat_name;
                top1_score = results[0].score;
            }
            bool correct = std::string(top1) == std::string(TEST_IMG_TRUE_LABEL);
            ESP_LOGI(TAG, "true=%s  pred=%s (%.3f)  %s",
                     TEST_IMG_TRUE_LABEL, top1, (double)top1_score,
                     correct ? "CORRECT" : "WRONG");
            for (size_t i = 0; i < results.size() && i < 3; i++)
                ESP_LOGI(TAG, "  top%u: %-22s %.4f", (unsigned)(i + 1),
                         results[i].cat_name, (double)results[i].score);
        } else {
            ESP_LOGI(TAG, "non-EuroSAT model (out_size=%d) — skip classification",
                     o->get_size());
        }

        // --- Phase 1: latency benchmark (parsed by bench.sh) ---
        // Re-running the same preprocessed input is a faithful, deterministic
        // inference (same MACs, same path) — ideal for timing and profiling.
        for (int i = 0; i < NUM_WARMUP; i++) model->run();

        int64_t total_us = 0, min_us = INT64_MAX, max_us = 0;
        for (int i = 0; i < NUM_TIMED; i++) {
            int64_t t0 = esp_timer_get_time();
            model->run();
            int64_t dt = esp_timer_get_time() - t0;
            total_us += dt;
            if (dt < min_us) min_us = dt;
            if (dt > max_us) max_us = dt;
        }
        double avg_ms = (double)total_us / NUM_TIMED / 1000.0;
        double cycles = (avg_ms / 1000.0) * CPU_FREQ_MHZ * 1e6;
        double mac_per_cycle = cycles > 0 ? (MODEL_MMAC * 1e6) / cycles : 0.0;

        // mJ/inf is measured externally (PPK2), so it is deliberately omitted here.
        snprintf(result_buf, sizeof(result_buf),
                 "OK %s ms/inf=%.2f MAC/cycle=%.4f min=%.2f max=%.2f "
                 "true=%s pred=%s score=%.3f",
                 MODEL_NAME, avg_ms, mac_per_cycle,
                 (double)min_us / 1000.0, (double)max_us / 1000.0,
                 TEST_IMG_TRUE_LABEL, top1, (double)top1_score);
        printf("STATUS err=none result=[%s]\n", result_buf);
        fflush(stdout);

        // --- Phase 2: power-profiling loop (continuous inference for PPK2) ---
        // Input tensor stays preprocessed; re-running it is a faithful, fully
        // deterministic inference (same MACs, same power) — ideal for profiling.
        int run_idx = 0;
        while (true) {
            run_idx++;
            for (int k = 1; k <= BURST_SIZE; k++) {
                int64_t t0 = esp_timer_get_time();
                model->run();
                int64_t t1 = esp_timer_get_time();
                printf("STATUS run=%d inf=%d ms/inf=%.2f true=%s pred=%s\n",
                       run_idx, k, (double)(t1 - t0) / 1000.0,
                       TEST_IMG_TRUE_LABEL, top1);
                fflush(stdout);
            }
            printf("STATUS run=%d burst_done\n", run_idx);
            fflush(stdout);
            vTaskDelay(pdMS_TO_TICKS(BURST_IDLE_MS));   // idle baseline window
        }
    }

    delete model;

idle:
    while (true) {
        printf("STATUS err=%s result=[%s]\n", err_msg, result_buf);
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
