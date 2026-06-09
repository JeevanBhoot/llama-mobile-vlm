// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#ifdef ANDROID

#include <jni.h>
#include <algorithm>
#include <atomic>
#include <fstream>

#include "squash.hpp"

namespace {
struct Session {
    squash::Model model;
    squash::Generator generator;
    explicit Session(squash::Model&& _model) : model(std::move(_model)), generator(model) {}
};
std::unique_ptr<Session> session;

struct Progress {
    static constexpr double None = -1.0;
    std::atomic<double> value{None};

    void set(double progress) {
        value.store(std::clamp(progress, 0.0, 1.0), std::memory_order_relaxed);
    }
    void clear() { value.store(None, std::memory_order_relaxed); }

    std::optional<double> get() const {
        auto progress = value.load(std::memory_order_relaxed);
        return progress < 0.0 ? std::nullopt : std::make_optional(progress);
    }
};
Progress progress;

struct ProgressScope {
    ProgressScope() { progress.set(0.0); }
    ~ProgressScope() { progress.clear(); }
};

// JNI helpers

struct StringHolder {
    JNIEnv* env;
    jstring jdata;
    const char* data;

    StringHolder(JNIEnv* env, jstring jdata)
        : env(env), jdata(jdata), data(env->GetStringUTFChars(jdata, nullptr)) {}
    StringHolder(const StringHolder&) = delete;
    StringHolder& operator=(const StringHolder&) = delete;
    ~StringHolder() { env->ReleaseStringUTFChars(jdata, data); }
};

std::optional<squash::Image> imageFromDirectBuffer(JNIEnv* env,
                                                   jint imageWidth,
                                                   jint imageHeight,
                                                   jobject imageData) {
    if (imageData == nullptr) {
        return std::nullopt;
    }
    if (imageWidth <= 0 || imageHeight <= 0) {
        throw std::runtime_error("Image width and height must be positive when image data is set");
    }

    auto* buffer = reinterpret_cast<uint8_t*>(env->GetDirectBufferAddress(imageData));
    auto capacity = env->GetDirectBufferCapacity(imageData);
    if (buffer == nullptr || capacity < 0) {
        throw std::runtime_error("Image data must be a direct ByteBuffer");
    }

    auto width = static_cast<size_t>(imageWidth);
    auto height = static_cast<size_t>(imageHeight);
    auto expectedSize = width * height * sizeof(uint32_t);
    if (static_cast<uint64_t>(capacity) != expectedSize) {
        std::ostringstream err;
        err << "Image buffer has size " << capacity << ", expected " << expectedSize;
        throw std::runtime_error(err.str());
    }

    auto* pixels = reinterpret_cast<uint32_t*>(buffer);
    std::vector<uint8_t> rgb(width * height * 3);
    for (auto i = size_t(0); i < width * height; ++i) {
        auto pixel = pixels[i];
        rgb[3 * i + 0] = static_cast<uint8_t>((pixel >> 16) & 0xff);
        rgb[3 * i + 1] = static_cast<uint8_t>((pixel >> 8) & 0xff);
        rgb[3 * i + 2] = static_cast<uint8_t>(pixel & 0xff);
    }

    return squash::Image(static_cast<uint>(imageHeight), static_cast<uint>(imageWidth),
                         std::move(rgb));
}

template <class T, class F>
T errorGuard(JNIEnv* env, F&& func) {
    try {
        return func();
    } catch (const std::exception& error) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), error.what());
        return T();
    }
}
}  // namespace

extern "C" JNIEXPORT void JNICALL  //
Java_ai_graphcore_squashedllama_Lib_load(JNIEnv* env, jobject /*this*/, jstring _path) {
    errorGuard<void>(env, [&] {
        ProgressScope _progressScope;
        squash::selectOmpNumThreads();
        StringHolder path(env, _path);
        session.reset();  // free memory before loading a new model
        std::ifstream file(path.data);
        session.reset(new Session{
            squash::loadSquashedTensors(file, [](double value) { progress.set(value); })});
    });
}

extern "C" JNIEXPORT void JNICALL  //
Java_ai_graphcore_squashedllama_Lib_unload(JNIEnv* env, jobject /*this*/) {
    errorGuard<void>(env, [&] { session.reset(); });
}

extern "C" JNIEXPORT jobject JNICALL  //
Java_ai_graphcore_squashedllama_Lib_progress(JNIEnv* env, jobject /*this*/) {
    auto current = progress.get();
    if (!current) {
        return nullptr;
    }
    auto doubleClass = env->FindClass("java/lang/Double");
    auto constructor = env->GetMethodID(doubleClass, "<init>", "(D)V");
    return env->NewObject(doubleClass, constructor, static_cast<jdouble>(*current));
}

extern "C" JNIEXPORT jobjectArray JNICALL  //
Java_ai_graphcore_squashedllama_Lib_prefill(JNIEnv* env,
                                            jobject /*this*/,
                                            jstring _prefix,
                                            jint imageWidth,
                                            jint imageHeight,
                                            jobject _imageData,
                                            jint maxGeneratedTokens,
                                            jdouble temperature,
                                            jint topK,
                                            jdouble topP) {
    return errorGuard<jobjectArray>(env, [&] {
        ProgressScope _progressScope;
        StringHolder prefix(env, _prefix);
        if (!session) {
            throw std::runtime_error("No model loaded");
        }
        auto image = imageFromDirectBuffer(env, imageWidth, imageHeight, _imageData);
        auto tokens = session->generator.prefill(prefix.data, std::move(image),
                                                 {.maxGeneratedTokens = uint(maxGeneratedTokens),
                                                  .seed = std::nullopt,
                                                  .temperature = float(temperature),
                                                  .topK = uint(topK),
                                                  .topP = float(topP)},
                                                 [](double value) { progress.set(value); });
        auto jarray =
            env->NewObjectArray(jsize(tokens.size()), env->FindClass("java/lang/String"), nullptr);
        for (auto i = 0u; i < tokens.size(); ++i) {
            auto jstring = env->NewStringUTF(tokens[i].c_str());
            env->SetObjectArrayElement(jarray, jsize(i), jstring);
            env->DeleteLocalRef(jstring);
        }
        return jarray;
    });
}

extern "C" JNIEXPORT jstring JNICALL  //
Java_ai_graphcore_squashedllama_Lib_generate(JNIEnv* env, jobject /*this*/) {
    return errorGuard<jstring>(env, [&] {
        if (!session) {
            throw std::runtime_error("No model loaded");
        }
        auto token = session->generator.generate();
        return env->NewStringUTF(token.c_str());
    });
}

#endif  // ANDROID
