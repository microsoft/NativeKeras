#pragma once

#include <string>
#include <vector>

#include "CNTKLibrary.h"

#include "Globals.h"
#include "DataBuffer.h"

#pragma warning (push)
#pragma warning (disable: 4251)
#pragma warning (disable: 4751)
#pragma warning (disable: 4800)
#include "KerasProto.pb.h"
#pragma warning (pop)

namespace keras
{
    namespace cntk_utils
    {
        class BufferMinibatchSource : public CNTK::MinibatchSource, public std::enable_shared_from_this<BufferMinibatchSource>
        {
        public:
            BufferMinibatchSource::BufferMinibatchSource(bool infinitelyRepeat = true, bool fullDataSweep = true)
                : mPos(0), mInfinitelyRepeat(infinitelyRepeat)
            {}

            void Add(const TensorProto & nda, const CNTK::NDShape & inputShape, const std::wstring name = L"");

            const std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData> & GetNextMinibatch(size_t batchSize, const CNTK::DeviceDescriptor & device = globals::device);

            const std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData>& GetNextMinibatch(
                size_t minibatchSizeInSequences,
                size_t minibatchSizeInSamples,
                size_t numberOfWorkers,
                size_t workerRank,
                const CNTK::DeviceDescriptor& device = globals::device);

            const std::wstring & LabelsName() const { return mInfos.back().m_name; }
            const std::wstring & FeaturesName() const { return mInfos.front().m_name; }

            const CNTK::StreamInformation & FeatureStreamInfo() const { return mInfos.front(); }
            const CNTK::StreamInformation & LabelStreamInfo() const { return mInfos.back(); }

            const std::unordered_set<CNTK::StreamInformation>& StreamInfos() { return mInfosSet; }

            size_t GetPos() const { return mPos; }
            size_t GetNumSamples() const { return mSamples; }

        private:
            std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData> mResult;
            std::unordered_set<CNTK::StreamInformation> mInfosSet;
            std::vector<CNTK::StreamInformation> mInfos;
            std::vector<NDArrayPtr> mArrays;
            std::vector<CNTK::NDShape> mInputShapes;
            size_t mPos;
            size_t mSamples;
            bool mInfinitelyRepeat;
        };
    }
}