# On-device inference library

## Development

```sh
./dev
./dev -p all tests
```

## Setup

```sh
./third-party/fetch.sh
sudo apt install ninja-build clang clang-format
# If android: install NDK to /opt/android-sdk/ndk/latest
```

## License information

 - Clang (compiler), Apache 2.0
 - Ninja (build system), Apache 2.0
 - Ninja utility `third-party/ninja_syntax.py`, Apache 2.0
 - Android NDK, [License](https://android.googlesource.com/platform/prebuilts/ndk/+/master/NOTICE)
 - nlohmann/json, MIT License
