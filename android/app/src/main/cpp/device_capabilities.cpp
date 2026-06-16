// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include <jni.h>

#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <cstdio>
#include <sys/auxv.h>
#endif

namespace {

#if defined(__aarch64__)
bool has(unsigned long value, unsigned long flag) {
    return (value & flag) != 0;
}

void appendMissing(char* buffer, size_t bufferSize, size_t* offset, const char* feature) {
    if (*offset >= bufferSize) return;
    if (*offset > 0) {
        *offset += std::snprintf(buffer + *offset, bufferSize - *offset, ", ");
    }
    if (*offset < bufferSize) {
        *offset += std::snprintf(buffer + *offset, bufferSize - *offset, "%s", feature);
    }
}

bool supportsRequiredFeatures() {
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    return has(hwcap, HWCAP_ASIMDDP) && has(hwcap2, HWCAP2_I8MM);
}
#endif

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_graphcore_llamamobiledemo_DeviceCapabilities_nativeSupportsRequiredCpuFeatures(
        JNIEnv*, jobject) {
#if defined(__aarch64__)
    return supportsRequiredFeatures() ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_graphcore_llamamobiledemo_DeviceCapabilities_nativeUnsupportedReason(
        JNIEnv* env, jobject) {
#if defined(__aarch64__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    char missing[64] = {};
    size_t offset = 0;
    if (!has(hwcap, HWCAP_ASIMDDP)) appendMissing(missing, sizeof(missing), &offset, "ASIMDDP");
    if (!has(hwcap2, HWCAP2_I8MM)) appendMissing(missing, sizeof(missing), &offset, "I8MM");

    if (missing[0] == '\0') {
        return env->NewStringUTF("Supported");
    }

    char reason[96] = {};
    std::snprintf(reason, sizeof(reason), "Missing CPU features: %s", missing);
    return env->NewStringUTF(reason);
#else
    return env->NewStringUTF("This build requires arm64-v8a.");
#endif
}
