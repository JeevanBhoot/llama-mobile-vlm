# On-Device Inference Library

This directory contains the "vanilla" C++ inference library, command-line demo, unit tests, and benchmarks for Llama-Mobile `.sqt` models.

See this guide for usage instructions, or jump straight into [`src/core/ops.cpp`](src/core/ops.cpp) to see the core operator implementations.


## Setup

```sh
./dev setup
sudo apt install clang clang-format gdb libomp-dev ninja-build
```

For Android builds, install the Android NDK at `/opt/android-sdk/ndk/latest`. Most commands can be run as `./dev -p android ...` to apply them to the Android target.


## Tests

```sh
./dev tests
```

Run all configured platforms, including Android when the NDK is available:

```sh
./dev -p all tests
```


## Prebuilt Models

Download the public release models:

```sh
mkdir -p models
aws s3 sync --no-sign-request \
  --region=eu-west-1 \
  s3://graphcore-research-public/2026-llama-mobile/models/20260611/ \
  models/
```


## CLI Generation

Text model:

```sh
echo "What is blue?" | ./dev run cli -- models/text-1B-int8.sqt -g 128
```

Vision-language model:

```sh
echo "Describe this image." | ./dev run cli -- models/vision-11B-s3d8.sqt -g 64 --image TEST_IMAGE.jpg
```

See also `./dev run cli -- --help`. The CLI reads one prompt per input line. Use `--benchmark` to save per-step timings to `cli.benchmark.jsonl`.


## Benchmarks

Build and run a model-shaped benchmark:

```sh
./dev run benchmark -- text_model_1B
```

Run selected lower-level benchmarks:

```sh
./dev run benchmark -- tensor_mlp
./dev run benchmark -- _tensor_matmulT_s3d8
```

Emit JSON lines to stdout and repeat each benchmark:

```sh
./dev run benchmark -- --json --repeat 5 text_model_1B
```

Benchmark prefixes select registered benchmark names. Public, paper-relevant
entry points include `text_model_1B`, `tensor_mlp`, and the `_tensor_*`
operator benchmarks. Names beginning with `_` are lower-level diagnostic
benchmarks and may be more hardware-specific.


## Android Target

Build and run the CLI on an attached Android device (check with `adb devices`):

```sh
adb push models/text-1B-int8.sqt /data/local/tmp/text-1B-int8.sqt
./dev -p android run cli -- text-1B-int8.sqt -g 16
```

The Android target is optimized for Armv9-A with BF16 and I8MM support. The
helper script pins threads on Pixel 8a devices; other devices run without that
device-specific taskset.


## Profiling Using `perf`

Install `linux-tools-generic`. On AWS Graviton, you may have to symlink the
matching `perf` binary into `/usr/local/bin`.

```conf
# Add to /etc/sysctl.conf, then restart
kernel.perf_event_paranoid = 0
```

```sh
./dev build benchmark
perf record -g ./build/release/benchmark text_model_1B
perf report
perf report -d benchmark
```


## VSCode C++ Configuration

<details markdown>

<summary>Example `.vscode/c_cpp_properties.json`</summary>

```json
{
    "configurations": [
        {
            "name": "linux-host",
            "cppStandard": "c++20",
            "intelliSenseMode": "linux-clang-x64",
            "compilerPath": "/usr/bin/clang++",
            "includePath": [
                "${workspaceFolder}/third-party",
                "${workspaceFolder}/src"
            ]
        },
        {
            "name": "android-arm64",
            "cppStandard": "c++20",
            "intelliSenseMode": "linux-clang-arm64",
            "compilerPath": "/opt/android-sdk/ndk/latest/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++",
            "compilerArgs": [
                "--target=aarch64-linux-android35",
                "-march=armv9-a+bf16+i8mm"
            ],
            "includePath": [
                "${workspaceFolder}/third-party",
                "${workspaceFolder}/src",
                "/opt/android-sdk/ndk/latest/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include",
                "/opt/android-sdk/ndk/latest/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/c++/v1"
            ],
            "defines": [
                "ANDROID"
            ]
        }
    ],
    "version": 4
}
```

</details>


## License Information

- Clang (compiler), Apache 2.0
- Ninja (build system), Apache 2.0
- Ninja utility `third-party/ninja_syntax.py`, Apache 2.0
- Android NDK, [License](https://android.googlesource.com/platform/prebuilts/ndk/+/master/NOTICE)
- C++ libraries:
  - nlohmann/json, MIT License
  - jarro2783/cxxopts, MIT License
  - catchorg/Catch2, Boost Software License 1.0
  - nothings/stb/{stb_image.h, stb_image_resize2.h}, Public Domain
