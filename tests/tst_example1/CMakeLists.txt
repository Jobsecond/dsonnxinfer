project(tst_example1 VERSION 0.0.0.1 LANGUAGES CXX)

file(GLOB_RECURSE _src *.h *.cpp)
add_executable(${PROJECT_NAME} ${_src})

find_package(syscmdline CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE dsonnxinfer::dsonnxinfer syscmdline::syscmdline)

target_include_directories(${PROJECT_NAME} PRIVATE ${DSONNXINFER_BUILD_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE .)

if (ONNXRUNTIME_ENABLE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONNXRUNTIME_ENABLE_CUDA)
endif()

if (ONNXRUNTIME_ENABLE_DML)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONNXRUNTIME_ENABLE_DML)
endif()
