// EuroSAT land-cover classifier on ESP32-S3 (Xtensa LX7) via ESP-DL.
// Follows Espressif's recommended classification flow:
//   ImagePreprocessor (resize + normalize + quantize-to-input-exponent)
//   -> Model::run()
//   -> ClsPostprocessor (dequantize + softmax + top-k argmax).
// Deploys MobileNetV3 (INT16, ~91.5% top-1). One embedded test image is
// classified and compared against its known label to verify the deploy path
// end-to-end (should match verify_accuracy.py's host prediction).
#include "dl_model_base.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_image_define.hpp"
#include "eurosat_cls_postprocessor.hpp"
#include "eurosat_category_name.hpp"
#include "test_image.h"
#include <cmath>

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstdio>
#include <string>

// Power-profiling protocol (read by espdl_bench/measure_power_ppk2.py):
//   STATUS run=<r> inf=<k> ms/inf=<t> true=<l> pred=<p>   (one per inference)
//   STATUS run=<r> burst_done                             (end of a burst)
// then BURST_IDLE_MS of idle so the host can sample an idle-baseline current.
#define BURST_SIZE     10        // inferences per run
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

        snprintf(result_buf, sizeof(result_buf),
                 "OK true=%s pred=%s score=%.3f",
                 TEST_IMG_TRUE_LABEL, top1, (double)top1_score);

        // --- power-profiling loop: bursts of timed inferences + idle gaps ---
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
