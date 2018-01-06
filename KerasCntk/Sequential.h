#pragma once

#include <string>

#include "CNTKLibrary.h"

#include "json.hpp"

#include "BufferMinibatchSource.h"
#include "CntkUtils.h"

namespace keras
{
    typedef void(__stdcall * ProgressCallback)(char * data, unsigned size);

    class Sequential
    {
    public:
        Sequential();

        void Init(const char * inData, unsigned inLen);

        void DriveCommand();

        void GetOutput(char ** outData, unsigned * outLen);

    public:
        void AddActivation(const nlohmann::json & jnode);

        void AddDense(const nlohmann::json & jnode);

        void AddConv1D(const nlohmann::json & jnode);
        void AddConv2D(const nlohmann::json &jnode);

        void AddMaxPooling1D(const nlohmann::json & jnode);
        void AddMaxPooling2D(const nlohmann::json & jnode);

        void AddAveragePooling1D(const nlohmann::json & jnode);
        void AddAveragePooling2D(const nlohmann::json & jnode);

        void AddGlobalMaxPooling1D(const nlohmann::json & jnode);

        void AddDropout(const nlohmann::json & jnode);
        void AddFlatten(const nlohmann::json & jnode);

        nlohmann::json AddEmbedding(const nlohmann::json & jnode);

    private:
        CNTK::ParameterInitializer CreateInitializer(const nlohmann::json & jnode);
        CNTK::FunctionPtr GetActivation(const nlohmann::json & jnode, const CNTK::Variable & operand);

        CNTK::Variable GetInputLayer(const nlohmann::json &jnode);

        CNTK::LearnerPtr CreateLearner(nlohmann::json & jnode);
        CNTK::FunctionPtr CreateLossFunction(nlohmann::json & jnode);
        void CreateErrorFunction(nlohmann::json & jnode);

        void CreateLabels(const std::wstring & name);

        nlohmann::json NodeOrNull(const nlohmann::json & jnode, const std::string & name);
        std::string GetOrCreateName(const nlohmann::json & jnode);

        void ParseFitParameters(const nlohmann::json & jnode);

        void SetupInputs();

        void Fit();
        void Predict();

        typedef std::unordered_map<std::string, double> HistoryValues;
        void UpdateProgress(HistoryCallbackType type, std::size_t id, const HistoryValues & historyValues);

        void InitProtoOutput(CNTK::DataType dataType = CNTK::DataType::Float, std::size_t nrows = 0, std::size_t ncols = 0);
        void AppendProtoOutput(const std::vector<std::vector<float>> & output);

        void Sequential::LoadModel();

    private:

        KerasProto _proto;

        CNTK::FunctionPtr _model;
        CNTK::LearnerPtr _learner;
        CNTK::FunctionPtr _loss;
        CNTK::FunctionPtr _error;

        CNTK::Variable _features;
        CNTK::Variable _labels;

        std::vector<CNTK::Variable> _inputVariables;

        // Fit parameters
        int _batchSize;
        int _nepochs;
        int _verbose;

        std::wstring _path;

        std::size_t _nsamples;

        bool _dataSource;

        std::shared_ptr<cntk_utils::BufferMinibatchSource> _bufferMinibatchSource;

        std::unordered_map<std::string, CNTK::FunctionPtr> _layersMap;
    };
}