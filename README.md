# ds-onnx-infer

DiffSinger inference utility

## Dependencies

+ qmsetup
+ flowonnx
+ syscmdline
+ nlohmann/json

## Setup Environment

```bash
git clone --recursive https://github.com/Jobsecond/dsonnxinfer.git

cd dsonnxinfer

pushd libs/flowonnx/libs
cmake [-Dep="gpu"|"gpu-cuda12"|"dml"] -P ../scripts/setup-onnxruntime.cmake
popd

git clone https://github.com/microsoft/vcpkg.git

pushd vcpkg
bootstrap-vcpkg.bat
vcpkg install --x-manifest-root=../scripts/vcpkg-manifest --x-install-root=./installed --triplet=x64-windows
popd
```

Configure CMake
```bash
-DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

-DDSONNXINFER_BUILD_STATIC:BOOL=OFF
-DDSONNXINFER_ENABLE_AUDIO_EXPORT:BOOL=ON
-DDSONNXINFER_BUILD_TESTS:BOOL=ON

# If using CUDA:
-DONNXRUNTIME_ENABLE_CUDA:BOOL=ON
# If using DirectML:
-DONNXRUNTIME_ENABLE_DML:BOOL=ON
```