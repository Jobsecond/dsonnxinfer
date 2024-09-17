# Add target

dsonnxinfer_add_library(${PROJECT_NAME} SHARED AUTOGEN)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(syscmdline CONFIG REQUIRED)

if(DEFINED DSONNXINFER_ENABLE_AUDIO_EXPORT AND DSONNXINFER_ENABLE_AUDIO_EXPORT)
    find_package(SndFile CONFIG REQUIRED)
    set(_audio_export_lib SndFile::sndfile)
    set(_audio_export_def DSONNXINFER_ENABLE_AUDIO_EXPORT)
endif()

# Configure target
file(GLOB_RECURSE _src *.h *.hpp *.cpp)
qm_configure_target(${PROJECT_NAME}
        SOURCES ${_src}
        FEATURES cxx_std_17
        LINKS_PRIVATE
            flowonnx::flowonnx
            nlohmann_json::nlohmann_json
            yaml-cpp::yaml-cpp
            syscmdline::syscmdline
            ${_audio_export_lib}
        INCLUDE_PRIVATE
        SYNC_INCLUDE_OPTIONS
        DEFINES
            ${_audio_export_def}
)

# Add install command
#ck_sync_include(${PROJECT_NAME})