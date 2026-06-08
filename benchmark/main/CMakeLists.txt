set(src_dirs ./)
set(include_dirs ./)
set(requires esp-tflite-micro esp_timer esp_psram heap)

set(embed_files "model.tflite")

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})

