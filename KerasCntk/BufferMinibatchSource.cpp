#include <codecvt>

#include "BufferMinibatchSource.h"

using namespace std;

namespace keras
{
    namespace cntk_utils
    {
        void BufferMinibatchSource::Add(const TensorProto & nda, const CNTK::NDShape & inputShape, const std::wstring name)
        {
            CNTK::NDShape shape(vector<size_t>(nda.shape().cbegin(), nda.shape().cend()));
            if (shape.SubShape(1).TotalSize() != inputShape.TotalSize())
                throw logic_error("The input shape is incompatible with the actual data shape");

            mInputShapes.push_back(inputShape);

            if (shape.TotalSize()*sizeof(float) > nda.data().size())
                throw logic_error("The shape is incompatible with the data size.");

            // std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> conversion;
            auto a = make_shared<DataBuffer>(shape, (float *)nda.data().data(), name);

            CNTK::StreamInformation si;
            si.m_elementType = a->DataType();
            si.m_sampleLayout = inputShape;
            si.m_storageFormat = CNTK::StorageFormat::Dense;
            if (name.size() == 0)
                si.m_name = CNTK::Internal::GenerateUid(L"StreamInformation");
            else
                si.m_name = name;

            if (mArrays.empty())
                mSamples = a->Shape()[0];
            else if (mSamples != a->Shape()[0])
                throw std::runtime_error("Inputs with different number of samples.");

            si.m_id = mInfos.size();

            // mData.insert({si, CNTK::MinibatchData(CNTK::MakeSharedObject<CNTK::Value>(a), a->Shape()[0])});
            mInfos.emplace_back(si);
            mInfosSet.emplace(si);
            mArrays.emplace_back(a);
        }

        const std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData> & BufferMinibatchSource::GetNextMinibatch(size_t batchSize, const CNTK::DeviceDescriptor & device)
        {
            bool eof;
            size_t nsamples;

            if (mSamples == mPos && mInfinitelyRepeat)
                mPos = 0;

            size_t left = mSamples - mPos;

            if (left == 0)
                throw logic_error("The batch is exhausted.");

            if (left <= batchSize)
            {
                nsamples = left;
                eof = true;
            }
            else
            {
                nsamples = batchSize;
                eof = false;
            }

            mResult.clear();
            for (auto i = 0; i < mArrays.size(); ++i)
            {
                const auto & nda = mArrays[i];
                CNTK::ValuePtr view = nda->GetBatch(mPos, mPos + nsamples, mInputShapes[i]);
                mResult.insert({ mInfos[i], CNTK::MinibatchData(view, nsamples, eof) });
            }

            mPos += nsamples;

            return mResult;
        }

        const std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData>& BufferMinibatchSource::GetNextMinibatch(
            size_t minibatchSizeInSequences,
            size_t minibatchSizeInSamples,
            size_t numberOfWorkers,
            size_t workerRank,
            const CNTK::DeviceDescriptor & device)
        {
            return GetNextMinibatch(minibatchSizeInSamples);
        }
    }
}