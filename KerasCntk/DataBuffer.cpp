#include <algorithm>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "fmt/format.h"

#include "DataBuffer.h"

using namespace std;

namespace keras
{
    namespace cntk_utils
    {
        DataBuffer::DataBuffer(const CNTK::NDShape & shape, const float * data, const std::wstring & name)
            : mShape(shape), 
              mDataType(CNTK::DataType::Float),
              mName(name),
              mDataTypeSize(sizeof(float)),
              mTransform(true),
              mPos(0),
              mFloatTensor(nullptr)
        {
            mFloatTensor = CreateFloatTensor(shape, data);
        }

        DataBuffer::DataBuffer(CNTK::DataType dataType, const CNTK::NDShape & shape, const std::wstring & name)
            : mShape(shape),
              mDataType(dataType),
              mName(name),
              mTransform(true),
              mPos(0),
              mFloatTensor(nullptr)
        {
            mDataTypeSize = dataType == CNTK::DataType::Double ? sizeof(double) : sizeof(float);
        }

        void DataBuffer::TransformIfNecessary(const CNTK::NDShape & shape)
        {
            if (!mTransform)
                return;

            // The tensor is transformed from [nsamples x data_shape] in row major order, into
            // [data_shape, 1, nsamples] in column major order. Row major order is the opposite
            // of column major order - we use only row major order, see the example below.
            //
            // Inputs:
            //      * Input shape: 200x300x3
            //      * Training data set: 1000x200x300
            //
            // In columnar order, the target is: 200x300x3x1x1000. The second last is the sequence
            // axis, the last - the batch axis.
            //
            // Thus, the target in row order is: 1000x1x3x300x200

            THLongStorage * newShape = THLongStorage_newWithSize(shape.Rank() + 2);
            for (int64_t i = 0; i < (int64_t)shape.Rank(); ++i)
                newShape->data[2 + i] = (long)shape[i];

            newShape->data[0] = (long)mShape[0];
            newShape->data[1] = 1;

            THFloatTensor * newTensor = THFloatTensor_newView(mFloatTensor, newShape);

            for (int i = 2, j = (int)newShape->size - 1; i < j; ++i, --j)
                THFloatTensor_transpose(newTensor, nullptr, i, j);

            // The data is actually re-arranged when we make the tensor contiguous
            THFloatTensor_free(mFloatTensor);
            mFloatTensor = THFloatTensor_newContiguous(newTensor);
            THFloatTensor_free(newTensor);

            mTransform = false;
        }

        CNTK::ValuePtr DataBuffer::GetBatch(size_t start, size_t end, const CNTK::NDShape & inputShape)
        {
            if(mDataType != CNTK::DataType::Float)
                throw runtime_error("Not implemented.");

            if (end > mShape[0])
                throw runtime_error(fmt::format("end [== {:d}] is out of range [max == {:d}]", end, mShape[0]));

            TransformIfNecessary(inputShape);

            THFloatTensor * view = THFloatTensor_newNarrow(mFloatTensor, 0, (long)start, (long)(end-start));
            view = THFloatTensor_newContiguous(view);
            size_t batchSize = 1;
            for (auto i = 0; i < view->nDimension; ++i)
                batchSize *= view->size[i];
            mFloats.resize(batchSize);
            memcpy(&mFloats[0], view->storage->data + view->storageOffset, batchSize*sizeof(float));

            return CNTK::Value::CreateBatch(inputShape, mFloats, globals::device, false);
        }

        void OutputBuffer::Add(const std::vector<std::vector<float>> & sequences)
        {
            for (const auto & seq : sequences)
                mFloatSequences.push_back(seq);
        }

        void OutputBuffer::GetTensorProto(TensorProto & proto)
        {
            size_t nrows;
            size_t ncols;
            if (mDataType == DataType::Float)
            {
                nrows = mFloatSequences.size();
                ncols = mFloatSequences[0].size();
            }

            proto.add_shape((int32_t)nrows);
            proto.add_shape((int32_t)ncols);
            proto.set_count((int32_t)(nrows*ncols));
                
            proto.set_type(mDataType);
            proto.set_format(TensorFormat::RowMajor);
            
            proto.mutable_data()->resize(proto.count() * mDataTypeSize);

            if (mDataType == DataType::Float)
            {
                char * p = &(*proto.mutable_data())[0];
                for (const auto & seq : mFloatSequences)
                {
                    size_t size = seq.size() * sizeof(float);
                    memcpy(p, &seq[0], size);
                    p += size;
                }
            }
        }
    }
}
