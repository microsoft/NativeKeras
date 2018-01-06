#include <iostream>

#include "gtest/gtest.h"

#include "TH/THTensor.h"

#include "Keras.h"
#include "DataBuffer.h"

using namespace std;
using namespace keras;

TEST(CntkFunctions, Argmax)
{
    // To warm up and understand how CNTK functions work. We
    // are not in the business of testing CNTK.

    // CNTK::SetTraceLevel(CNTK::TraceLevel::Info);

    auto device = CNTK::DeviceDescriptor::CPUDevice();

    vector<float> data = { 0.35f, 0.4f, 0.5f, 0.4f };
    CNTK::Variable var = CNTK::InputVariable(CNTK::NDShape({ data.size() }), CNTK::DataType::Float, "PredVar");
    auto view = CNTK::MakeSharedObject<CNTK::NDArrayView>(CNTK::NDShape({ data.size() }), data, true);
    auto val = CNTK::MakeSharedObject<CNTK::Value>(view);

    auto func = CNTK::Argmax(var, CNTK::Axis::AllStaticAxes());

    auto resultShape = func->Output().Shape();
    resultShape = resultShape.AppendShape({ 1, 1 });

    vector<float> resultData(resultShape.TotalSize());
    CNTK::ValuePtr resultVal = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(resultShape, resultData, false));
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> resultMap = { { func->Output(), resultVal } };

    func->Evaluate({ {var, val} }, resultMap);

    vector<float> expected = { 2.0f };
    ASSERT_EQ(resultData.size(), expected.size());
    for (auto i = 0; i < resultData.size(); ++i)
        ASSERT_EQ(resultData[i], expected[i]);
}

TEST(KerasFunctions, CategoricalAccuracy)
{
    // CNTK::SetTraceLevel(CNTK::TraceLevel::Info);

    auto device = CNTK::DeviceDescriptor::CPUDevice();

    vector<float> predData = { 0.35f, 0.4f, 0.5f, 0.4f };
    vector<float> truthData = { 1.1f, 1.3f, 1.0f, 4.0f };

    CNTK::Variable predVar = CNTK::InputVariable(CNTK::NDShape({ predData.size() }), CNTK::DataType::Float, "PredVar");
    CNTK::Variable truthVar = CNTK::InputVariable(CNTK::NDShape({ truthData.size() }), CNTK::DataType::Float, "TruthVar");

    auto predView = CNTK::MakeSharedObject<CNTK::NDArrayView>(CNTK::NDShape({ predData.size() }), predData, false);
    auto truthView = CNTK::MakeSharedObject<CNTK::NDArrayView>(CNTK::NDShape({ truthData.size() }), truthData, false);

    auto predVal = CNTK::MakeSharedObject<CNTK::Value>(predView);
    auto truthVal = CNTK::MakeSharedObject<CNTK::Value>(truthView);

    auto func = keras::CategoricalAccuracy(predVar, truthVar, CNTK::Axis::AllStaticAxes());

    auto resultVar = func->Output();

    auto resultShape = func->Output().Shape();
    resultShape = resultShape.AppendShape({ 1, 1 });

    vector<float> resultData(resultShape.TotalSize());
    CNTK::ValuePtr resultVal = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(resultShape, resultData, false));
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> resultMap = { { func->Output(), resultVal } };

    func->Evaluate({ { predVar, predVal }, { truthVar, truthVal } }, resultMap);

    vector<float> expected = { 0.0f };
    ASSERT_EQ(resultData.size(), expected.size());
    for (auto i = 0; i < resultData.size(); ++i)
        ASSERT_EQ(resultData[i], expected[i]);

    truthData = { 1.1f, 1.3f, 6.0f, 4.0f };

    func->Evaluate({ { predVar, predVal },{ truthVar, truthVal } }, resultMap);

    expected = { 1.0f };
    ASSERT_EQ(resultData.size(), expected.size());
    for (auto i = 0; i < resultData.size(); ++i)
        ASSERT_EQ(resultData[i], expected[i]);
}

TEST(KerasFunctions, CategoricalCrossEntropy)
{
    // CNTK::SetTraceLevel(CNTK::TraceLevel::Info);

    auto device = CNTK::DeviceDescriptor::CPUDevice();

    vector<float> predData = { 1.0f, 2.0f, 11.0f, 4.0f };
    vector<float> truthData = { 0.0f, 0.0f, 1.0f, 0.0f };

    CNTK::Variable predVar = CNTK::InputVariable(CNTK::NDShape({ predData.size() }), CNTK::DataType::Float, "PredVar");
    CNTK::Variable truthVar = CNTK::InputVariable(CNTK::NDShape({ truthData.size() }), CNTK::DataType::Float, "TruthVar");

    auto predView = CNTK::MakeSharedObject<CNTK::NDArrayView>(CNTK::NDShape({ predData.size() }), predData, false);
    auto truthView = CNTK::MakeSharedObject<CNTK::NDArrayView>(CNTK::NDShape({ truthData.size() }), truthData, false);

    auto predVal = CNTK::MakeSharedObject<CNTK::Value>(predView);
    auto truthVal = CNTK::MakeSharedObject<CNTK::Value>(truthView);

    auto func = keras::CategoricalCrossEntropy(predVar, truthVar);

    auto resultVar = func->Output();

    auto resultShape = func->Output().Shape();
    resultShape = resultShape.AppendShape({ 1, 1 });

    vector<float> resultData(resultShape.TotalSize());
    CNTK::ValuePtr resultVal = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(resultShape, resultData, false));
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> resultMap = { { func->Output(), resultVal } };

    func->Evaluate({ { predVar, predVal },{ truthVar, truthVal } }, resultMap);

    vector<float> expected = { 0.4925f };
    ASSERT_EQ(resultData.size(), expected.size());
    for (auto i = 0; i < resultData.size(); ++i)
        ASSERT_NEAR(resultData[i], expected[i], 0.0001);
}

static inline THLongStorage * CreateLongStorage(const vector<int> & shape)
{
    THLongStorage * storage = THLongStorage_newWithSize(shape.size());
    for (auto i = 0; i < shape.size(); ++i)
        storage->data[i] = shape[i];
    return storage;
}

TEST(Tensor, Permute)
{
    vector<int> shape = { 600, 23, 25, 3 };
    THLongStorage * storage = CreateLongStorage(shape);
    ASSERT_EQ(storage->size, shape.size());
    auto totalSize = storage->data[0] * storage->data[1] * storage->data[2] * storage->data[3];
    THFloatTensor * t1 = THFloatTensor_newWithSize(storage, nullptr);
    ASSERT_EQ(t1->storage->size, totalSize);
    for (auto i = 0; i < totalSize; ++i)
        t1->storage->data[i] = (float)i;

    THFloatTensor * t2 = THFloatTensor_newClone(t1);

    THFloatTensor_transpose(t2, nullptr, 0, 3);
    THFloatTensor_transpose(t2, nullptr, 1, 2);

    ASSERT_FALSE(THFloatTensor_isContiguous(t2));

    t2 = THFloatTensor_newContiguous(t2);

    ASSERT_TRUE(THFloatTensor_isContiguous(t2));

    for (auto i = 0; i < shape[0]; ++i)
    for (auto j = 0; j < shape[1]; ++j)
    for (auto k = 0; k < shape[2]; ++k)
    for (auto l = 0; l < shape[3]; ++l)
    {
        ASSERT_EQ(THFloatTensor_get4d(t1, i, j, k, l), THFloatTensor_get4d(t2, l, k, j, i));
    }
}

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();

    return 0;
}