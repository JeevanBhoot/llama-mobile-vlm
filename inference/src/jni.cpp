#ifdef ANDROID

#include <jni.h>
#include <fstream>

#include "squash.hpp"

namespace {
struct Session {
    squash::Model model;
    squash::Generator generator;
    explicit Session(squash::Model&& _model) : model(std::move(_model)), generator(model) {}
};
std::unique_ptr<Session> session;

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
        squash::selectOmpNumThreads();
        StringHolder path(env, _path);
        session.reset();  // free memory before loading a new model
        std::ifstream file(path.data);
        session.reset(new Session{squash::loadSquashedTensors(file)});
    });
}

extern "C" JNIEXPORT void JNICALL  //
Java_ai_graphcore_squashedllama_Lib_unload(JNIEnv* env, jobject /*this*/) {
    errorGuard<void>(env, [&] { session.reset(); });
}

extern "C" JNIEXPORT jobjectArray JNICALL  //
Java_ai_graphcore_squashedllama_Lib_prefill(JNIEnv* env,
                                            jobject /*this*/,
                                            jstring _prefix,
                                            jint maxGeneratedTokens,
                                            jdouble temperature,
                                            jint topK,
                                            jdouble topP) {
    return errorGuard<jobjectArray>(env, [&] {
        StringHolder prefix(env, _prefix);
        if (!session) {
            throw std::runtime_error("No model loaded");
        }
        auto tokens =
            session->generator.prefill(prefix.data, {.maxGeneratedTokens = uint(maxGeneratedTokens),
                                                     .seed = std::nullopt,
                                                     .temperature = float(temperature),
                                                     .topK = uint(topK),
                                                     .topP = float(topP)});
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
