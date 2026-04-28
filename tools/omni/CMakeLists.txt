# omni

find_package(Threads REQUIRED)

# CoreML / ANE support option
option(ENABLE_COREML "Enable CoreML/ANE support for vision encoder" OFF)
set(OMNI_TTS_BACKEND "legacy" CACHE STRING "Omni TTS backend: legacy or voxcpm")
set_property(CACHE OMNI_TTS_BACKEND PROPERTY STRINGS legacy voxcpm)
if (NOT OMNI_TTS_BACKEND STREQUAL "legacy" AND NOT OMNI_TTS_BACKEND STREQUAL "voxcpm")
    message(FATAL_ERROR "OMNI_TTS_BACKEND must be one of: legacy, voxcpm")
endif()

set(OMNI_TTS_BACKEND_SOURCES tts_backend.h)
if (OMNI_TTS_BACKEND STREQUAL "legacy")
    list(APPEND OMNI_TTS_BACKEND_SOURCES tts_backend_legacy.cpp)
else()
    set(VOXCPM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../VoxCPM.cpp" CACHE PATH "Path to VoxCPM.cpp source tree")
    if (NOT EXISTS "${VOXCPM_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "OMNI_TTS_BACKEND=voxcpm requires VOXCPM_SOURCE_DIR to point at VoxCPM.cpp")
    endif()
    set(VOXCPM_USE_EXTERNAL_GGML ON CACHE BOOL "Use llama.cpp ggml target for VoxCPM" FORCE)
    set(VOXCPM_BUILD_EXAMPLES OFF CACHE BOOL "Build VoxCPM examples" FORCE)
    set(VOXCPM_BUILD_BENCHMARK OFF CACHE BOOL "Build VoxCPM benchmarks" FORCE)
    set(VOXCPM_BUILD_TESTS OFF CACHE BOOL "Build VoxCPM tests" FORCE)
    set(VOXCPM_EXTERNAL_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml" CACHE PATH "Parent ggml source for VoxCPM" FORCE)
    set(VOXCPM_USE_EXTERNAL_MINIAUDIO ON CACHE BOOL "Use omni-linked miniaudio symbols for VoxCPM" FORCE)
    add_subdirectory("${VOXCPM_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/voxcpm")
    target_include_directories(voxcpm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor")
    list(APPEND OMNI_TTS_BACKEND_SOURCES
        tts_backend_voxcpm.cpp
        "${VOXCPM_SOURCE_DIR}/third_party/miniaudio/miniaudio.c")
endif()

add_library(omni
            omni.cpp
            omni.h
            audition.cpp
            audition.h
            vision.cpp
            vision.h
            omni-impl.h
            token2wav/token2wav-impl.cpp
            token2wav/token2wav-impl.h
            token2wav/token2wav.cpp
            token2wav/token2wav.h
            ${OMNI_TTS_BACKEND_SOURCES}
            )

# Add CoreML related files when enabled
if(ENABLE_COREML)
    target_sources(omni PRIVATE
        coreml/omni_coreml.h
        coreml/omni_coreml.mm
        coreml/coreml_minicpmo45_vit_all_f16.h
        coreml/coreml_minicpmo45_vit_all_f16.m
    )
    # Define compile-time macro for code guards
    target_compile_definitions(omni PRIVATE ENABLE_COREML)

    # Enable ARC for Objective-C files
    set_source_files_properties(coreml/omni_coreml.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(coreml/coreml_minicpmo45_vit_all_f16.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

target_link_libraries     (omni PUBLIC ggml llama common)
target_link_libraries     (omni PRIVATE Threads::Threads)
target_include_directories(omni PUBLIC  .)
target_include_directories(omni PRIVATE ../..)
target_include_directories(omni PRIVATE ../../vendor)
target_compile_features   (omni PRIVATE cxx_std_17)

if (OMNI_TTS_BACKEND STREQUAL "legacy")
    target_compile_definitions(omni PUBLIC OMNI_TTS_BACKEND_LEGACY=1)
else()
    target_compile_definitions(omni PUBLIC OMNI_TTS_BACKEND_VOXCPM=1)
    target_link_libraries(omni PUBLIC voxcpm)
endif()

# Link CoreML and Accelerate frameworks when CoreML is enabled
if(ENABLE_COREML)
    target_link_libraries(omni PRIVATE
        "-framework Foundation"
        "-framework CoreML"
        "-framework Accelerate"
        "-ObjC"
    )
endif()

# Windows-specific definitions
if (WIN32)
    target_compile_definitions(omni PRIVATE NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (omni PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(omni PRIVATE LLAMA_BUILD)
    target_compile_definitions(omni PUBLIC  LLAMA_SHARED)
    if (WIN32)
        set_target_properties(omni PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

set(OMNI_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/omni.h
    )

set_target_properties(omni
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS omni LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(omni PRIVATE -Wno-cast-qual)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(omni        BUILD_INFO)
endif()

set(TARGET llama-omni-cli)
add_executable         (${TARGET} omni-cli.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-omni-cli)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# 单工模式 - 纯音频测试（批量 prefill + 单次 decode）
set(TARGET_SINGLE_TEST_AUDIO llama-omni-single-test-audio)
add_executable         (${TARGET_SINGLE_TEST_AUDIO} test/single_test_audio.cpp)
set_target_properties  (${TARGET_SINGLE_TEST_AUDIO} PROPERTIES OUTPUT_NAME llama-omni-single-test-audio)
target_link_libraries  (${TARGET_SINGLE_TEST_AUDIO} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_SINGLE_TEST_AUDIO} PRIVATE cxx_std_17)

# 单工模式 - Omni (audio+vision) 测试
set(TARGET_SINGLE_TEST_OMNI llama-omni-single-test-omni)
add_executable         (${TARGET_SINGLE_TEST_OMNI} test/single_test_omni.cpp)
set_target_properties  (${TARGET_SINGLE_TEST_OMNI} PROPERTIES OUTPUT_NAME llama-omni-single-test-omni)
target_link_libraries  (${TARGET_SINGLE_TEST_OMNI} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_SINGLE_TEST_OMNI} PRIVATE cxx_std_17)

# 双工模式测试
set(TARGET_DUPLEX_TEST llama-omni-test-duplex)
add_executable         (${TARGET_DUPLEX_TEST} test/test-duplex.cpp)
set_target_properties  (${TARGET_DUPLEX_TEST} PROPERTIES OUTPUT_NAME llama-omni-test-duplex)
target_link_libraries  (${TARGET_DUPLEX_TEST} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_DUPLEX_TEST} PRIVATE cxx_std_17)

# # Token2Wav 测试程序 - 完整 C++ 语音合成
# add_executable(token2wav-example token2wav/token2wav-example.cpp)
# target_link_libraries(token2wav-example PRIVATE common omni Threads::Threads)
# target_compile_features(token2wav-example PRIVATE cxx_std_17)

