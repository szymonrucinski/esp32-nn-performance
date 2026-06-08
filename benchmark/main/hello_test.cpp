#include <cstdio>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "hello";

extern "C" void app_main(void)
{
    while (true) {
        ESP_LOGI(TAG, "Hello from ESP32-S3! Tick...");
        printf("printf: hello\n");
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
