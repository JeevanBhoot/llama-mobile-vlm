#ifdef ANDROID

#include <jni.h>
#include "squash.hpp"

extern "C" JNIEXPORT jint JNICALL Java_ai_graphcore_squashedllama_Lib_meaning(JNIEnv* /* env */,
                                                                              jobject /* this */) {
    return 42;
}

#endif  // ANDROID
