LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := device_capabilities
LOCAL_SRC_FILES := device_capabilities.cpp
LOCAL_CPPFLAGS := -std=c++17 -Wall -Wextra -Werror -march=armv8-a
LOCAL_LDFLAGS := -Wl,-z,max-page-size=16384
include $(BUILD_SHARED_LIBRARY)
