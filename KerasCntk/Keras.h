#pragma once

#include <string>

#include "CNTKLibrary.h"

#include "json.hpp"

#ifdef KERAS_EXPORTS
#define KERAS_API __declspec(dllexport)
#else
#define KERAS_API __declspec(dllimport)
#endif // KERAS_EXPORTS

namespace keras
{
    KERAS_API CNTK::FunctionPtr CategoricalCrossEntropy(const CNTK::Variable & prediction, const CNTK::Variable & targets, const std::wstring & name = L"");
    KERAS_API CNTK::FunctionPtr CategoricalAccuracy(const CNTK::Variable & prediction, const CNTK::Variable & targets, const CNTK::Axis & axis, const std::wstring & name = L"");
    KERAS_API CNTK::FunctionPtr Embedding(const CNTK::Variable & prediction, size_t embeddingDim, const CNTK::Axis & axis, const std::wstring & name = L"");
}
