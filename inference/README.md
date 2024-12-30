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
sudo apt install ninja-build clang clang-format
# If android: install NDK to /opt/android-sdk/ndk/latest
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
