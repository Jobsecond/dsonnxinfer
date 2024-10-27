# ds-onnx-infer

DiffSinger inference utility

## Dependencies

+ qmsetup
+ flowonnx
  + onnxruntime
+ syscmdline
+ nlohmann/json

## Setup Environment

### Step 1: Clone repository
```bash
git clone --recursive https://github.com/Jobsecond/dsonnxinfer.git

cd dsonnxinfer
```

### Step 2: Download onnxruntime
#### (1) Change directory to flowonnx libs

```bash
pushd libs/flowonnx/libs
# On Windows use backslash:
# pushd libs\flowonnx\libs
```

#### (2) Download ONNX Runtime
* For CUDA 11.x version:
   ```bash
   cmake -Dep="gpu" -P ../scripts/setup-onnxruntime.cmake
   ```
* For CUDA 12.x version:
   ```bash
   cmake -Dep="gpu-cuda12" -P ../scripts/setup-onnxruntime.cmake
   ```
* For DirectML:
   ```bash
   cmake -Dep="dml" -P ../scripts/setup-onnxruntime.cmake
   ```
* For CPU only:
   ```bash
   cmake -P ../scripts/setup-onnxruntime.cmake
   ```

#### (3) Return to dsonnxinfer repo root directory:
```bash
popd
```

### Step 3: Download other dependencies
```bash
git clone https://github.com/microsoft/vcpkg.git

pushd vcpkg
bootstrap-vcpkg.bat
vcpkg install --x-manifest-root=../scripts/vcpkg-manifest --x-install-root=./installed --triplet=x64-windows
popd
```

List of vcpkg triplets:
* Windows MSVC: `x64-windows`
* Windows MinGW (NOT recommended): `x64-mingw-dynamic`
* Linux: `x64-linux`
* macOS (Apple Silicon): `arm64-osx`
* macOS (Intel): `x64-osx`

### CMake Configure Options
Add these to CMake configure options:
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

<details>
<summary>For MinGW users:</summary>
If you are using MinGW on Windows, you need to install the vcpkg packages using <code>x64-mingw-dynamic</code> triplet, and specify the vcpkg triplet in CMake configure options:

```bash
-DVCPKG_TARGET_TRIPLET=x64-mingw-dynamic
```

Please note that MinGW support is currently experimental and is NOT recommended to use in production. Windows users are encouraged to use MSVC.
</details>
