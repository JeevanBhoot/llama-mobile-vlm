// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

package ai.graphcore.llamamobiledemo

object DeviceCapabilities {
    private val loadError: Throwable? = try {
        System.loadLibrary("device_capabilities")
        null
    } catch (throwable: Throwable) {
        throwable
    }

    fun supportsRequiredCpuFeatures(): Boolean {
        return loadError == null && nativeSupportsRequiredCpuFeatures()
    }

    fun unsupportedReason(): String {
        return loadError?.message?.let { "Could not load device capability checker: $it" }
            ?: nativeUnsupportedReason()
    }

    private external fun nativeSupportsRequiredCpuFeatures(): Boolean
    private external fun nativeUnsupportedReason(): String
}
