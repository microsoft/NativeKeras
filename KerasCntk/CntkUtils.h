#pragma once

#include "CNTKLibrary.h"

#include "Globals.h"
#include "DataBuffer.h"

namespace keras
{
    namespace cntk_utils
    {
        class HistoryAccumulator : public CNTK::ProgressWriter
        {
        public:
            HistoryAccumulator()
                : CNTK::ProgressWriter(1, 0, 1, 0, 1, 0)
            { }

            void OnWriteTrainingUpdate(
                const std::pair<size_t, size_t> & samples,
                const std::pair<size_t, size_t> & updates,
                const std::pair<double, double> & aggregateLoss,
                const std::pair<double, double> & aggregateMetric)
            {
                mLoss.push_back(aggregateLoss.second);
                mAcc.push_back(aggregateMetric.second);
                mSamples.push_back(samples.second);
                mUpdates.push_back(updates.second);
            }

            double AvgLoss() const
            {
                if (mSamples.size() == 0) return 0.0;
                return (double)mLoss.back() / mSamples.back();
            }

            double AvgAcc() const
            {
                if (mSamples.size() == 0) return 0.0;
                return (double)mAcc.back() / mSamples.back();
            }

        private:
            std::vector<double> mLoss;
            std::vector<double> mAcc;
            std::vector<size_t> mSamples;
            std::vector<size_t> mUpdates;
        };

        CNTK::FunctionPtr LoadModel(const char * buffer, size_t len);
    }
}