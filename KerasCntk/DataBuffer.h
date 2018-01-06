#pragma once

#include "TH/THTensor.h"

#include <string>
#include <vector>

#include "CNTKLibrary.h"

#pragma warning (push)
#pragma warning (disable: 4251)
#pragma warning (disable: 4751)
#pragma warning (disable: 4800)
#include "KerasProto.pb.h"
#pragma warning (pop)

#include "Keras.h"
#include "Globals.h"

namespace keras
{
    namespace cntk_utils
    {
        class DataBuffer : public std::enable_shared_from_this<DataBuffer>
        {
        public:
            KERAS_API DataBuffer(const CNTK::NDShape & shape, const float * data, const std::wstring & name = L"");
            KERAS_API DataBuffer(CNTK::DataType dataType, const CNTK::NDShape & shape, const std::wstring & name = L"");

            ~DataBuffer()
            {
                if (mFloatTensor != nullptr)
                    THFloatTensor_free(mFloatTensor);
            }

            KERAS_API CNTK::ValuePtr GetBatch(size_t start, size_t end, const CNTK::NDShape & inputShape);

            const CNTK::NDShape & Shape() const { return mShape; }
            const CNTK::DataType DataType() const { return mDataType; }

            size_t CalcSampleSize(size_t start, size_t batchSize) const
            {
                size_t left = mShape[0] - start;
                return std::min(batchSize, left);
            }

        private:
            void TransformIfNecessary(const CNTK::NDShape & shape);

            bool mTransform;

            CNTK::NDShape mShape;
            CNTK::DataType mDataType;
            int mDataTypeSize;

            std::vector<uint8_t> mBatch;

            std::vector<float> mFloats;
            std::vector<double> mDoubles;

            THFloatTensor * mFloatTensor;
            size_t mPos;

            std::wstring mName;
        };

        typedef std::shared_ptr<DataBuffer> NDArrayPtr;

        class OutputBuffer
        {
        public:
            OutputBuffer(CNTK::DataType dataType = globals::dataType)
                : mDataType(dataType == CNTK::DataType::Double ? DataType::Double : DataType::Float),
                  mDataTypeSize(dataType == CNTK::DataType::Double ? 8 : 4)
            { }

            void Add(const std::vector<std::vector<float>> & sequences);
            void GetTensorProto(TensorProto & proto);

        private:
            DataType mDataType;
            size_t mDataTypeSize;
            std::vector<std::vector<float>> mFloatSequences;
            std::vector<std::vector<double>> mDoubleSequences;
        };

        inline THLongStorage * CreateLongStorage(const CNTK::NDShape & shape)
        {
            THLongStorage * storage = THLongStorage_newWithSize(shape.Rank());
            for (auto i = 0; i < shape.Rank(); ++i)
                storage->data[i] = (long)shape[i];
            return storage;
        }

        inline THFloatTensor * CreateFloatTensor(const CNTK::NDShape & shape, const float * data = nullptr)
        {
            auto storage = CreateLongStorage(shape);
            THFloatTensor * tensor = THFloatTensor_newWithSize(storage, nullptr);
            if(data != nullptr)
                memcpy(tensor->storage->data, data, shape.TotalSize() * sizeof(float));
            THLongStorage_free(storage);
            return tensor;
        }

        inline std::string FloatTensorShapeAsString(THFloatTensor * tensor)
        {
            std::string result;
            for (auto i = 0; i < tensor->nDimension; ++i)
            {
                if (i > 0)
                    result += "x";
                result += std::to_string(tensor->size[i]);
            }
            return result;
        }
    }
}