# On-device inference library

## Development

```sh
./dev
./dev -p all tests

# Try out a language model
./dev run cli ../models/Llama-3.2-1B-Instruct-BF16.sqt
```

## Setup

```sh
./dev setup
sudo apt install clang clang-format gdb libc++-dev libopenmp-dev ninja-build
# If android: install NDK to /opt/android-sdk/ndk/latest
```

### Profiling using perf

Install `linux-tools-generic`. On AWS Graviton, you may have to `sudo ln -s /usr/lib/linux-tools-6.8.0-85/perf /usr/local/bin/perf` and `rm /usr/bin/perf`.

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

## License information

 - Clang (compiler), Apache 2.0
 - Ninja (build system), Apache 2.0
 - Ninja utility `third-party/ninja_syntax.py`, Apache 2.0
 - Android NDK, [License](https://android.googlesource.com/platform/prebuilts/ndk/+/master/NOTICE)
 - C++ Libraries
   - nlohmann/json, MIT License
   - jarro2783/cxxopts, MIT License
   - catchorg/Catch2, Boost Software License 1.0
   - nothings/stb/{stb_image.h, stb_image_resize2.h}, Public Domain
