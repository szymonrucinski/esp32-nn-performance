set(src_dirs ./)
set(include_dirs ./)
set(requires esp-dl esp_timer esp_psram heap)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

target_add_aligned_binary_data(${COMPONENT_LIB} "${CMAKE_CURRENT_SOURCE_DIR}/model.espdl" BINARY)
