#include <cstdio>
#include <cstring>
#include "model_config.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "benchmark";

extern const uint8_t model_tflite_start[] asm("_binary_model_tflite_start");
extern const uint8_t model_tflite_end[] asm("_binary_model_tflite_end");

#define NUM_WARMUP 2
#define NUM_RUNS 5
#define CPU_FREQ_MHZ 240
#define EST_POWER_MW 50.0f

static const char *err_msg = "none";
static char result_buf[256] = "no results yet";

extern "C" void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(5000));

    size_t model_size = model_tflite_end - model_tflite_start;
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    const tflite::Model *model = tflite::GetModel(model_tflite_start);
    if (!model) { err_msg = "GetModel null"; goto idle; }
    if (model->version() != TFLITE_SCHEMA_VERSION) { err_msg = "schema mismatch"; goto idle; }

    {
        size_t free_psram_now = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        size_t kArenaSize = (free_psram_now > 512 * 1024) ? free_psram_now - 256 * 1024 : 256 * 1024;
        uint8_t *tensor_arena = (uint8_t *)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM);
        if (!tensor_arena) { err_msg = "arena alloc fail"; goto idle; }

        tflite::MicroMutableOpResolver<30> resolver;
        resolver.AddAdd();
        resolver.AddCast();
        resolver.AddConcatenation();
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddFullyConnected();
        resolver.AddHardSwish();
        resolver.AddLogistic();
        resolver.AddMaxPool2D();
        resolver.AddMean();
        resolver.AddMul();
        resolver.AddPack();
        resolver.AddPad();
        resolver.AddPadV2();
        resolver.AddRelu();
        resolver.AddReshape();
        resolver.AddShape();
        resolver.AddSlice();
        resolver.AddStridedSlice();
        resolver.AddTranspose();

        tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);
        if (interpreter.AllocateTensors() != kTfLiteOk) {
            err_msg = "AllocateTensors fail";
            heap_caps_free(tensor_arena);
            goto idle;
        }

        TfLiteTensor *input = interpreter.input(0);
        memset(input->data.f, 0, input->bytes);

        for (int i = 0; i < NUM_WARMUP; i++) {
            TfLiteStatus inv_status = interpreter.Invoke();
            if (inv_status != kTfLiteOk) {
                static char inv_err[64];
                snprintf(inv_err, sizeof(inv_err), "warmup invoke fail i=%d status=%d", i, inv_status);
                err_msg = inv_err;
                heap_caps_free(tensor_arena);
                goto idle;
            }
        }

        int64_t total_us = 0;
        int64_t min_us = INT64_MAX;
        int64_t max_us = 0;
        for (int i = 0; i < NUM_RUNS; i++) {
            int64_t start = esp_timer_get_time();
            interpreter.Invoke();
            int64_t elapsed = esp_timer_get_time() - start;
            total_us += elapsed;
            if (elapsed < min_us) min_us = elapsed;
            if (elapsed > max_us) max_us = elapsed;
        }

        float avg_ms = (float)total_us / NUM_RUNS / 1000.0f;
        float mj_per_inf = EST_POWER_MW * avg_ms / 1000.0f;
        float total_cycles = (avg_ms / 1000.0f) * CPU_FREQ_MHZ * 1e6f;
        float mac_per_cycle = (total_cycles > 0) ? (MODEL_MMAC * 1e6f) / total_cycles : 0;

        snprintf(result_buf, sizeof(result_buf),
                 "OK %s ms/inf=%.2f mJ/inf=%.2f MAC/cycle=%.4f min=%.2f max=%.2f",
                 MODEL_NAME, (double)avg_ms, (double)mj_per_inf, (double)mac_per_cycle,
                 (double)(min_us / 1000.0f), (double)(max_us / 1000.0f));

        heap_caps_free(tensor_arena);
    }

idle:
    while (true) {
        printf("STATUS err=%s psram=%u internal=%u model=%u result=[%s]\n",
               err_msg, (unsigned)free_psram, (unsigned)free_internal,
               (unsigned)(model_tflite_end - model_tflite_start), result_buf);
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
