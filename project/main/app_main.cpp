#include "bsp/esp-bsp.h"
#include "dl_image_jpeg.hpp"
#include "esp_log.h"
#include "imagenet_cls.hpp"
#include "lvgl.h"
#include <cstdio>

extern const uint8_t cat_jpg_start[] asm("_binary_cat_jpg_start");
extern const uint8_t cat_jpg_end[] asm("_binary_cat_jpg_end");
const char *TAG = "mobilenetv2_cls";

static lv_obj_t *s_result_label = nullptr;

static void ui_init(void)
{
    lv_display_t *disp = bsp_display_start();   // init ST7789 LCD + LVGL port (LVGL v9)
    if (disp == nullptr) {
        ESP_LOGE(TAG, "bsp_display_start failed");
        return;
    }
    bsp_display_backlight_on();

    // Must hold the LVGL port mutex around all lv_* calls. Use a real timeout.
    if (!bsp_display_lock(1000)) {
        ESP_LOGE(TAG, "display lock failed");
        return;
    }

    lv_obj_t *scr = lv_screen_active();
    lv_obj_set_style_bg_color(scr, lv_color_black(), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, LV_PART_MAIN);

    lv_obj_t *title = lv_label_create(scr);
    lv_obj_set_style_text_font(title, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_set_style_text_color(title, lv_color_hex(0x00FF88), LV_PART_MAIN);
    lv_label_set_text(title, "testing mobilnet");
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 20);

    s_result_label = lv_label_create(scr);
    lv_obj_set_style_text_font(s_result_label, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_obj_set_style_text_color(s_result_label, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_text_align(s_result_label, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    lv_label_set_text(s_result_label, "running\ninference...");
    lv_obj_align(s_result_label, LV_ALIGN_CENTER, 0, 10);

    bsp_display_unlock();
}

static void ui_set_result(const char *text)
{
    if (s_result_label == nullptr) {
        return;
    }
    if (bsp_display_lock(1000)) {
        lv_label_set_text(s_result_label, text);
        bsp_display_unlock();
    }
}

extern "C" void app_main(void)
{
    ui_init();

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)cat_jpg_start,
                                      .data_len = (size_t)(cat_jpg_end - cat_jpg_start)};
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    ImageNetCls *cls = new ImageNetCls();
    auto &results = cls->run(img);

    char buf[128] = "no result";
    bool first = true;
    for (const auto &res : results) {
        ESP_LOGI(TAG, "category: %s, score: %f", res.cat_name, res.score);
        if (first) {
            snprintf(buf, sizeof(buf), "%s\n%.1f%%", res.cat_name, res.score * 100.0f);
            first = false;
        }
    }

    ui_set_result(buf);

    delete cls;
    heap_caps_free(img.data);
}
