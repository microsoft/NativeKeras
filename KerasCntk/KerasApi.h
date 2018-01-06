#pragma once

#include <cstdint>

extern "C" __declspec(dllimport) void KerasFitModel(
    const char * inData, unsigned inLen,
    char ** outData, unsigned * outLen,
    uint64_t * outPtr);