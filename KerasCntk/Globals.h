#pragma once

#include "CNTKLibrary.h"

namespace keras
{
    namespace globals
    {
        static const char * CHANNELS_LAST = "channels_last";
        static const char * CHANNELS_FIRST = "channels_first";

        static CNTK::DataType dataType = CNTK::DataType::Float;
        static double epsilon = FLT_EPSILON;
        static CNTK::DeviceDescriptor device = CNTK::DeviceDescriptor::CPUDevice();
        static std::string dataFormat = CHANNELS_LAST;
    }
}