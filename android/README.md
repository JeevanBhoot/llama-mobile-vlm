# Android Demo

This guide includes instructions for building the APK for yourself.

You can find our build at: https://graphcore-research-public.s3.eu-west-1.amazonaws.com/2026-llama-mobile/demo.apk

## Native Library

Build the JNI inference library from the repository root:

```sh
cd ../inference
./dev -p android build
cd ../android
```

The APK packages it via the symlink at `app/src/main/jniLibs/arm64-v8a/libsquashed-llama.so`.

## Build

If using the Android Studio JDK:

```sh
export JAVA_HOME=/opt/android-studio/jbr
```

Debug build/install:

```sh
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

## Release Build

Set release signing Gradle properties, for example in `~/.gradle/gradle.properties`:

```properties
LLAMA_MOBILE_KEYSTORE=/path/to/release.jks
LLAMA_MOBILE_KEYSTORE_PASSWORD=...
LLAMA_MOBILE_KEY_ALIAS=...
LLAMA_MOBILE_KEY_PASSWORD=...
```

Build:

```sh
./gradlew :app:assembleRelease
```

Publish:

```sh
aws s3 cp app/build/outputs/apk/release/app-release.apk s3://graphcore-research-public/2026-llama-mobile/demo.apk
```
