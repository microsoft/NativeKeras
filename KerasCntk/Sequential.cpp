#include <codecvt>
#include <cstdint>
#include <exception>
#include <fstream>
#include <unordered_map>

// TH headers [before anything else to avoid conflicts]
#include "TH/THTensor.h"

// CNTK headers
#include "CNTKLibrary.h"

// Third party headers
#include "fmt/format.h"
#include "json.hpp"

// Keras headers
#include "CntkUtils.h"
#include "Globals.h"
#include "Utils.h"

#pragma warning (push)
#pragma warning (disable: 4251)
#pragma warning (disable: 4751)
#pragma warning (disable: 4800)
#include "KerasProto.pb.h"
#pragma warning (pop)

#include "BufferMinibatchSource.h"
#include "DataBuffer.h"
#include "Sequential.h"

using namespace std;
using namespace nlohmann;
namespace cntk = CNTK;

namespace keras
{
    static ProgressCallback gProgressCallback = nullptr;

    typedef unordered_map<string, CNTK::FunctionPtr> ModelCache;

    ModelCache gModelCache;

    Sequential::Sequential()
        : _dataSource(false)
    {
        _bufferMinibatchSource = make_shared<cntk_utils::BufferMinibatchSource>();
    }

    void Sequential::Init(const char * inData, unsigned inLen)
    {
        _proto.ParseFromArray(inData, inLen);

        if (_proto.batch_size() > 0)
            _batchSize = _proto.batch_size();
        else
            _batchSize = 32;

        if (_proto.epochs() > 0)
            _nepochs = _proto.epochs();
        else
            _nepochs = 10;

        _verbose = 1;

        if (_proto.nsamples() > 0)
            _nsamples = (size_t)_proto.nsamples();

        _path = utils::ToWide(_proto.path());

        switch (_proto.verbose())
        {
        case 0:
            cntk::SetTraceLevel(cntk::TraceLevel::Error);
            break;

        case 1:
            cntk::SetTraceLevel(cntk::TraceLevel::Warning);
            break;

        default:
            cntk::SetTraceLevel(cntk::TraceLevel::Info);
            break;
        }

        gProgressCallback = (ProgressCallback)_proto.progress_callback();
    }

    void Sequential::DriveCommand()
    {
        switch (_proto.command())
        {
        case KerasCommand::Fit:
            Fit();
            break;

        case KerasCommand::Predict:
            Predict();
            break;
        }
    }

    void Sequential::GetOutput(char ** outData, unsigned * outLen)
    {
        string buffer;
        _proto.SerializeToString(&buffer);
        *outData = new char[buffer.size()];
        memcpy(*outData, buffer.data(), buffer.size());
        *outLen = (unsigned)buffer.size();
    }

    json Sequential::NodeOrNull(const json & jnode, const string & name)
    {
        auto it = jnode.find(name);
        if (it == jnode.end())
            return json::value_t::null;
        return *it;
    }

    string Sequential::GetOrCreateName(const nlohmann::json & jnode)
    {
        auto it = jnode.find("name");
        if (it == jnode.end())
            return utils::GenerateUuid();
        return it->get<string>();
    }

    cntk::ParameterInitializer Sequential::CreateInitializer(const json & jnode)
    {
        // The default
        if (jnode.is_null())
            return cntk::GlorotUniformInitializer();

        if (jnode.is_string())
        {
            auto str = jnode.get<string>();
            if (str == "glorot_uniform")
                return cntk::GlorotUniformInitializer();
            else if (str == "constant")
                return cntk::ConstantInitializer();
            else if (str == "zeros")
                return cntk::ConstantInitializer(0.0);
            else if (str == "ones")
                return cntk::ConstantInitializer(1.0);
            else if (str == "glorot_normal")
                return cntk::GlorotNormalInitializer();
            else if (str == "random_normal")
                return cntk::NormalInitializer(0.05);
            else if (str == "truncated_normal")
                return cntk::TruncatedNormalInitializer(0.05);

            throw logic_error("'" + str + "' initializer is not supported [yet].");
        }

        if (jnode.is_object())
        {
            auto op = jnode.at("op").get<string>();
            if (op == "glorot_uniform")
                return cntk::GlorotUniformInitializer(cntk::DefaultParamInitScale, cntk::SentinelValueForInferParamInitRank, cntk::SentinelValueForInferParamInitRank, jnode.value<unsigned long>("seed", cntk::SentinelValueForAutoSelectRandomSeed));
            if (op == "glorot_normal")
                return cntk::GlorotNormalInitializer(cntk::DefaultParamInitScale, cntk::SentinelValueForInferParamInitRank, cntk::SentinelValueForInferParamInitRank, jnode.value<unsigned long>("seed", cntk::SentinelValueForAutoSelectRandomSeed));
            else if (op == "constant")
                return cntk::ConstantInitializer(jnode.value<double>("value", 0.0));
            else if (op == "zeros")
                return cntk::ConstantInitializer(0.0);
            else if (op == "ones")
                return cntk::ConstantInitializer(1.0);
            else if (op == "random_normal")
                return cntk::NormalInitializer(jnode.value<double>("scale", 0.05), cntk::SentinelValueForInferParamInitRank, cntk::SentinelValueForInferParamInitRank, jnode.value<unsigned long>("seed", cntk::SentinelValueForAutoSelectRandomSeed));
            else if (op == "truncated_normal")
                return cntk::TruncatedNormalInitializer(jnode.value<double>("scale", 0.05), jnode.value<unsigned long>("seed", cntk::SentinelValueForAutoSelectRandomSeed));

            throw logic_error("'" + op + "' initializer is not supported [yet].");
        }

        throw logic_error("Bad initializer '" + jnode.dump() + "'");
    }

    cntk::FunctionPtr Sequential::GetActivation(const json & jnode, const cntk::Variable & operand)
    {
        if (jnode.is_string())
        {
            auto str = jnode.get<string>();
            if (str == "softmax")
                return cntk::Softmax(operand);
            else if (str == "elu")
                return cntk::ELU(operand);
            else if (str == "relu")
                return cntk::ReLU(operand);
            else if (str == "tanh")
                return cntk::Tanh(operand);
            else if (str == "sigmoid")
                return cntk::Sigmoid(operand);

            throw logic_error("'" + str + "' activation is not supported.");
        }

        if (jnode.is_object())
        {
            auto str = jnode.at("activation").get<string>();
            if (str == "softmax")
                return cntk::Softmax(operand);
            else if (str == "elu")
                return cntk::ELU(operand);
            else if (str == "relu")
                return cntk::ReLU(operand);
            else if (str == "tanh")
                return cntk::Tanh(operand);
            else if (str == "sigmoid")
                return cntk::Sigmoid(operand);

            throw logic_error("'" + str + "' activation is not supported.");
        }
        throw logic_error("Bad activation '" + jnode.dump() + "'");
    }

    cntk::Variable Sequential::GetInputLayer(const json &jnode)
    {
        cntk::Variable input;
        if (_model != nullptr)
        {
            input = _model;
        }
        else
        {
            auto it = jnode.find("input_shape");
            if (it == jnode.end())
                throw runtime_error("input_shape missing in the first network layer");
            cntk::NDShape ndshape(jnode.at("input_shape").get<vector<size_t>>());
            // mFeatures = cntk::InputVariable(ndshape, Globals::dataType, L"Features", { cntk::Axis::DefaultBatchAxis() });
            _features = cntk::InputVariable(ndshape, globals::dataType, L"Features");
            input = _features;
            _inputVariables.push_back(_features);
        }
        return input;
    }

    void Sequential::AddDense(const json & jnode)
    {
        size_t units = jnode.at("units").get<size_t>();
        bool useBias = jnode.value("use_bias", true);

        cntk::Variable input = GetInputLayer(jnode);

        auto kernelInitializer = CreateInitializer(NodeOrNull(jnode, "kernel_initializer"));
        auto timesParam = cntk::Parameter({ units, input.Shape()[0] }, globals::dataType, kernelInitializer);
        auto resultFunc = cntk::Times(timesParam, input);

        if (useBias)
        {
            auto biasInitializer = CreateInitializer(NodeOrNull(jnode, "bias_initializer"));
            auto plusParam = cntk::Parameter({ units }, globals::dataType, biasInitializer);
            resultFunc = cntk::Plus(plusParam, resultFunc);
        }

        auto activation = NodeOrNull(jnode, "activation");
        if (!activation.is_null())
        {
            resultFunc = GetActivation(activation, resultFunc);
        }

        _model = resultFunc;
    }

    void Sequential::AddConv1D(const json & jnode)
    {
        bool useBias = jnode.value("use_bias", false);

        cntk::Variable input = GetInputLayer(jnode);

        auto it = jnode.find("filters");
        if (it == jnode.end())
            throw runtime_error("filters missing in the first network layer");

        it = jnode.find("kernel_size");
        if (it == jnode.end())
            throw runtime_error("kernel_size missing in the first network layer");

        if (input.Shape().Rank() > 2)
            throw runtime_error("Conv1D's input is either 1- or 2-dimensional");

        size_t nfilters = jnode.at("filters").get<size_t>();
        cntk::NDShape kernelShape = { jnode.at("kernel_size").get<size_t>() };

        std::vector<size_t> strides1D = { jnode.at("strides").get<size_t>() };
        cntk::NDShape strides(strides1D);

        // The convolution is really slow on CPU without full sharing.
        vector<bool> sharing = { true };

        auto padding = jnode.value<string>("padding", "valid");

        auto kernelInitializer = cntk::GlorotUniformInitializer();
        auto convolutionParam = cntk::Parameter({ kernelShape[0], nfilters }, globals::dataType, kernelInitializer);
        auto resultFunc = cntk::Convolution(convolutionParam, input, strides, sharing, { padding == "same" }, { 1 }, 0);

        // wcout << "shape: " << resultFunc->Output().Shape().AsString() << endl;

        if (useBias)
        {
            auto biasInitializer = CreateInitializer(NodeOrNull(jnode, "bias_initializer"));
            auto plusParam = cntk::Parameter({ resultFunc->Output().Shape()[0] }, globals::dataType, biasInitializer);
            resultFunc = cntk::Plus(plusParam, resultFunc);
        }

        auto activation = NodeOrNull(jnode, "activation");
        if (!activation.is_null())
        {
            resultFunc = GetActivation(activation, resultFunc);
        }

        _model = resultFunc;
    }

    void Sequential::AddConv2D(const json & jnode)
    {
        bool useBias = jnode.value("use_bias", false);

        cntk::Variable input = GetInputLayer(jnode);

        auto it = jnode.find("filters");
        if (it == jnode.end())
            throw runtime_error("filters missing in the first network layer");

        it = jnode.find("kernel_size");
        if (it == jnode.end())
            throw runtime_error("kernel_size missing in the first network layer");

        // TODO [ivpop] Implement Keras' logic for channels first/last
        size_t nchannels = input.Shape()[input.Shape().Rank() - 1];
        size_t nfilters = jnode.at("filters").get<size_t>();
        cntk::NDShape kernelShape(jnode.at("kernel_size").get<vector<size_t>>());

        auto strides2D = jnode.at("strides").get<vector<size_t>>();
        strides2D.push_back(nchannels);
        cntk::NDShape strides(strides2D);

        // The convolution is really slow on CPU without full sharing.
        vector<bool> sharing = { true };

        auto padding = jnode.value<string>("padding", "valid");

        auto kernelInitializer = cntk::GlorotUniformInitializer();
        auto convolutionParam = cntk::Parameter({ kernelShape[0], kernelShape[1], nchannels, nfilters }, globals::dataType, kernelInitializer);
        auto resultFunc = cntk::Convolution(convolutionParam, input, strides, sharing, { padding == "same", padding == "same", false });

        if (useBias)
        {
            auto biasInitializer = CreateInitializer(NodeOrNull(jnode, "bias_initializer"));
            auto plusParam = cntk::Parameter({ 1, 1, nfilters }, globals::dataType, biasInitializer);
            resultFunc = cntk::Plus(plusParam, resultFunc);
        }

        auto activation = NodeOrNull(jnode, "activation");
        if (!activation.is_null())
        {
            resultFunc = GetActivation(activation, resultFunc);
        }

        _model = resultFunc;
    }

    void Sequential::AddMaxPooling1D(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);

        size_t pool = jnode.at("pool_size").get<size_t>();
        auto j = NodeOrNull(jnode, "strides");
        size_t strides = j.is_null() ? pool : jnode.at("strides").get<size_t>();
        auto padding = utils::ToLower(jnode.value<string>("padding", "valid")) == "same";
        _model = cntk::Pooling(_model, cntk::PoolingType::Max, { pool, 1 }, { strides, 1 }, { padding });
    }

    void Sequential::AddGlobalMaxPooling1D(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);
        _model = cntk::ReduceMax(input, cntk::Axis(1));
    }

    void Sequential::AddMaxPooling2D(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);

        cntk::NDShape pool(jnode.at("pool_size").get<vector<size_t>>());
        cntk::NDShape strides(jnode.at("strides").get<vector<size_t>>());

        auto padding = utils::ToLower(jnode.value<string>("padding", "valid")) == "same";
        _model = cntk::Pooling(_model, cntk::PoolingType::Max, { pool[0], pool[1], 1 }, { strides[0], strides[1], 1 }, { padding });
    }

    void Sequential::AddAveragePooling1D(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);

        size_t pool = jnode.at("pool_size").get<size_t>();
        auto j = NodeOrNull(jnode, "strides");
        size_t strides = j.is_null() ? pool : jnode.at("strides").get<size_t>();
        auto padding = utils::ToLower(jnode.value<string>("padding", "valid")) == "same";
        _model = cntk::Pooling(_model, cntk::PoolingType::Average, { pool, 1 }, { strides, 1 }, { padding });
    }

    void Sequential::AddAveragePooling2D(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);

        cntk::NDShape pool(jnode.at("pool_size").get<vector<size_t>>());
        cntk::NDShape strides(jnode.at("strides").get<vector<size_t>>());

        auto padding = utils::ToLower(jnode.value<string>("padding", "valid")) == "same";
        _model = cntk::Pooling(_model, cntk::PoolingType::Average, { pool[0], pool[1], 1 }, { strides[0], strides[1], 1 }, { padding });
    }

    void Sequential::AddActivation(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);
        _model = GetActivation(jnode, input);
    }

    void Sequential::AddDropout(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);
        auto seed = jnode.value<unsigned long>("seed", cntk::SentinelValueForAutoSelectRandomSeed);
        _model = cntk::Dropout(input, jnode.at("rate").get<double>(), seed);
    }

    void Sequential::AddFlatten(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);
        vector<size_t> dims;
        dims.push_back(input.Shape().TotalSize());
        cntk::NDShape new_shape(dims);
        _model = cntk::Reshape(input, new_shape);
    }

    json Sequential::AddEmbedding(const json & jnode)
    {
        cntk::Variable input = GetInputLayer(jnode);

        // Currently we don't use input_dim. We obtain the input dimension from the
        // input variable. Unclear yet what is the role of input_dim in Keras.
        size_t inputDim = jnode.at("input_dim").get<size_t>();
        size_t outputDim = jnode.at("output_dim").get<size_t>();

        auto it = jnode.find("input_length");
        if (it != jnode.end())
            throw runtime_error("input_length is not supported yet in the embedding layer");
        // size_t inputLength = it->get<size_t>();

        _features = cntk::InputVariable({ input.Shape()[0] }, globals::dataType, L"Features");
        _inputVariables.push_back(_features);

        string name = GetOrCreateName(jnode);

        auto embeddingParameters = cntk::Parameter({ outputDim, input.Shape()[0] }, globals::dataType, cntk::GlorotUniformInitializer(), globals::device);
        _model = cntk::Times(embeddingParameters, _features, utils::ToWide(name));

        _layersMap[name] = _model;

        json result;
        result["id"] = name;
        return result;
    }

    cntk::LearnerPtr Sequential::CreateLearner(json & jnode)
    {
        auto optimizer = jnode["optimizer"];

        string optimizerString;

        if (optimizer.is_object())
            optimizerString = utils::ToLower(optimizer.at("op").get<string>());
        else if (optimizer.is_string())
            optimizerString = utils::ToLower(optimizer.get<string>());
        else
            throw logic_error("Bad optimizer '" + jnode.dump() + "'");

        if (optimizerString == "sgd")
        {
            double learningRate = 0.01;
            double momentum = 0.0;
            double decay = 0.0;
            bool nestorov = false;

            cntk::AdditionalLearningOptions options;

            if (optimizer.is_object())
            {
                learningRate = optimizer.value<double>("lr", 0.01);
                momentum = optimizer.value<double>("momentum", 0.0);
                decay = optimizer.value<double>("decay", 0.0);
                nestorov = optimizer.value<bool>("nestorov", false);
            }

            auto lrs = cntk::LearningRateSchedule(learningRate);

            auto result = cntk::SGDLearner(_model->Parameters(), lrs, options);

            return result;
        }
        else if (optimizerString == "adadelta")
        {
            double learningRate = 0.01;
            double rho = 0.95;
            double epsilon = 1e-08;
            double decay = 0.0;

            cntk::AdditionalLearningOptions options;

            if (optimizer.is_object())
            {
                learningRate = optimizer.value<double>("lr", 0.01);
                rho = optimizer.value<double>("rho", 0.95);
                epsilon = optimizer.value<double>("epsilon", 1e-08);
                decay = optimizer.value<double>("decay", 0.0);
            }

            auto lrs = cntk::LearningRateSchedule(learningRate);

            auto result = cntk::AdaDeltaLearner(_model->Parameters(), lrs, rho, epsilon, options);

            return result;
        }
        else
        {
            throw logic_error("'" + optimizerString + "' optimizer is not supported [yet].");
        }
    }

    void Sequential::CreateLabels(const wstring & name = L"Labels")
    {
        // The labels are the last input variable
        if (_proto.inputs_size() > 0)
        {
            _dataSource = false;
            auto id = _proto.inputs_size() - 1;
            CNTK::NDShape shape(std::vector<size_t>(_proto.inputs().Get(id).shape().cbegin(), _proto.inputs().Get(id).shape().cend()));
            _labels = cntk::InputVariable(shape.SubShape(1), globals::dataType, name);
        }
        else
        {
            _dataSource = true;
            _labels = cntk::InputVariable({ _proto.nlabels() }, globals::dataType, name);
            _nsamples = _proto.nsamples();
        }
        _inputVariables.push_back(_labels);
    }

    __declspec(dllexport) cntk::FunctionPtr CategoricalCrossEntropy(const cntk::Variable & prediction, const cntk::Variable & targets, const wstring& name)
    {
        cntk::FunctionPtr result = cntk::ReduceSum(prediction, cntk::Axis::AllStaticAxes());
        result = cntk::ElementDivide(prediction, result);
        result = cntk::Clip(result, cntk::Constant::Scalar((float)globals::epsilon), cntk::Constant::Scalar(1.0f - (float)globals::epsilon));
        result = cntk::ElementTimes(targets, cntk::Log(result));
        return cntk::Minus(cntk::Constant::Scalar(0.0f), cntk::ReduceSum(result, cntk::Axis::AllStaticAxes()));
    }

    __declspec(dllexport) cntk::FunctionPtr CategoricalAccuracy(const cntk::Variable & prediction, const cntk::Variable & targets, const cntk::Axis & axis, const wstring& name)
    {
        return cntk::Equal(cntk::Argmax(targets, axis), cntk::Argmax(prediction, axis));
    }

    void Sequential::CreateErrorFunction(json & jnode)
    {
        for (json::iterator it = jnode["metrics"].begin(); it != jnode["metrics"].end(); ++it)
        {
            auto & jobj = *it;

            if (jobj.get<string>() == "accuracy")
                _error = CategoricalAccuracy(_model, _labels, cntk::Axis::AllStaticAxes());
        }
    }

    cntk::FunctionPtr Sequential::CreateLossFunction(json & jnode)
    {
        auto jloss = jnode["loss"];

        string lossString;

        if (jloss.is_string())
            lossString = jloss.get<string>();
        else
            throw logic_error("Bad optimizer '" + jnode.dump() + "'");

        cntk::FunctionPtr result = nullptr;

        if (lossString == "categorical_crossentropy")
            result = CategoricalCrossEntropy(_model, _labels);
        else if (lossString == "binary_crossentropy")
            result = CNTK::BinaryCrossEntropy(_model, _labels);
        else
            throw logic_error("Bad loss function '" + lossString + "'");

        return result;
    }

    void Sequential::ParseFitParameters(const json & jnode)
    {
        _batchSize = jnode.value<int>("batch_size", 32);
        _nepochs = jnode.value<int>("epochs", 10);
        _verbose = jnode.value<int>("verbose", 1);
    }

    void Sequential::SetupInputs()
    {
        if (_proto.inputs_size() > 0)
        {
            for (auto i = 0; i < _proto.inputs_size(); ++i)
                _bufferMinibatchSource->Add(_proto.inputs().Get(i), _inputVariables.at(i).Shape());
            _nsamples = _proto.inputs().Get(0).shape().Get(0);
        }
        else
        {
            _dataSource = true;
        }
    }

    void Sequential::UpdateProgress(HistoryCallbackType type, size_t id, const HistoryValues & historyValues)
    {
        if (gProgressCallback == nullptr)
            return;

        HistoryProto proto;
        for (const auto & kv : historyValues)
        {
            proto.add_names(kv.first);
            proto.add_values(kv.second);
        }
        proto.set_id((unsigned)id);
        proto.set_type(type);

        string buffer;
        proto.SerializeToString(&buffer);
        gProgressCallback(&buffer[0], (unsigned)buffer.size());
    }

    void Sequential::Fit()
    {
        cntk::DeviceDescriptor::TrySetDefaultDevice(globals::device);

        auto jroot = json::parse(_proto.graph().c_str());

        // The operators are under the graph element
        for (json::iterator it = jroot["graph"].begin(); it != jroot["graph"].end(); ++it)
        {
            auto & jnode = *it;

            auto op = utils::ToLower(jnode["op"].get<string>());

            if (op == "dense")
                AddDense(jnode);
            else if (op == "conv2d")
                AddConv2D(jnode);
            else if (op == "conv1d")
                AddConv1D(jnode);
            else if (op == "activation")
                AddActivation(jnode);
            else if (op == "dropout")
                AddDropout(jnode);
            else if (op == "flatten")
                AddFlatten(jnode);
            else if (op == "maxpooling1d")
                AddMaxPooling1D(jnode);
            else if (op == "maxpooling2d")
                AddMaxPooling2D(jnode);
            else if (op == "averagepooling1d")
                AddAveragePooling1D(jnode);
            else if (op == "averagepooling2d")
                AddAveragePooling2D(jnode);
            else if (op == "embedding")
                AddEmbedding(jnode);
        }

        CreateLabels();

        // The compile parameters
        auto jnode = jroot["compile_params"];
        _learner = CreateLearner(jnode);
        _loss = CreateLossFunction(jnode);
        CreateErrorFunction(jnode);

        // The trainer
        auto history = make_shared<cntk_utils::HistoryAccumulator>();
        auto trainer = cntk::CreateTrainer(_model, _loss, _error, { _learner }, { history });

        jnode = NodeOrNull(jroot, "fit_params");
        if (!jnode.is_null())
            ParseFitParameters(jnode);

        CNTK::StreamInformation featureStreamInfo;
        CNTK::StreamInformation labelStreamInfo;

        // The minibatch source
        SetupInputs();
        CNTK::MinibatchSourcePtr minibatchSource;
        if (_dataSource)
        {
            auto featureStreamName = L"features";
            auto labelsStreamName = L"labels";

            minibatchSource = cntk::TextFormatMinibatchSource(
                _path,
                { { featureStreamName, _features.Shape().TotalSize(), true },
                { labelsStreamName, _labels.Shape().TotalSize(), true } },
                _nsamples,
                false);

            featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
            labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);
        }
        else
        {
            featureStreamInfo = _bufferMinibatchSource->FeatureStreamInfo();
            labelStreamInfo = _bufferMinibatchSource->LabelStreamInfo();

            minibatchSource = _bufferMinibatchSource;
        }

        size_t numBatchesToTrain = (_nepochs * _nsamples) / _batchSize;
        size_t outputFrequencyInBatches = numBatchesToTrain;

        UpdateProgress(HistoryCallbackType::TrainingBegin, 0, {});

        double trainLossValue = 0.0;
        double evaluationValue = 0.0;

        double trainingSamples = 0.0;
        double epochSamples = 0.0;
        double batchSamples = 0.0;

        for (size_t epoch = 0; epoch < _nepochs; ++epoch)
        {
            epochSamples = 0.0;

            UpdateProgress(HistoryCallbackType::EpochBegin, epoch, { { "acc", evaluationValue },{ "loss", trainLossValue } });

            for (size_t samples = 0, batchId = 0; samples < _nsamples; ++batchId)
            {
                UpdateProgress(HistoryCallbackType::BatchBegin, batchId, { { "acc", evaluationValue },{ "loss", trainLossValue } });

                auto minibatchData = minibatchSource->GetNextMinibatch(_batchSize, globals::device);
                trainer->TrainMinibatch({ { _features, minibatchData[featureStreamInfo] },{ _labels, minibatchData[labelStreamInfo] } }, globals::device);
                double trainLossValue = trainer->PreviousMinibatchLossAverage();
                double evaluationValue = trainer->PreviousMinibatchEvaluationAverage();

                batchSamples = (double)trainer->PreviousMinibatchSampleCount();
                samples += (unsigned long)trainer->PreviousMinibatchSampleCount();
                epochSamples += batchSamples;
                trainingSamples += batchSamples;

                UpdateProgress(HistoryCallbackType::BatchEnd, batchId, { { "acc", evaluationValue },{ "loss", trainLossValue },{ "nsamples", batchSamples } });
            }

            UpdateProgress(HistoryCallbackType::EpochEnd, epoch, { { "acc", evaluationValue },{ "loss", trainLossValue },{ "nsamples", epochSamples } });
        }

        UpdateProgress(HistoryCallbackType::TrainingEnd, 0, { { "acc", evaluationValue },{ "loss", trainLossValue },{ "nsamples", trainingSamples } });

        // Serialize the model. Not many options but to write to a file and load the file.
        auto tempPath = tmpnam(nullptr);
        _model->Save(utils::ToWide(tempPath));

        ifstream inStream(tempPath, ifstream::ate | ifstream::binary);
        auto fileSize = inStream.tellg();
        _proto.mutable_model()->resize(fileSize);
        inStream.seekg(ifstream::beg);
        inStream.read(&(*_proto.mutable_model())[0], fileSize);
        _unlink(tempPath);
    }

    void Sequential::InitProtoOutput(CNTK::DataType dataType, size_t nrows, size_t ncols)
    {
        _proto.mutable_outputs()->Clear();
        auto o = _proto.mutable_outputs()->Add();
        o->mutable_data()->reserve(nrows*ncols*(dataType == CNTK::DataType::Double ? 8 : 4));
        o->add_shape(0);
        o->add_shape((int32_t)ncols);
        o->set_format(TensorFormat::RowMajor);
        o->set_type(dataType == CNTK::DataType::Double ? DataType::Double : DataType::Float);
    }

    void Sequential::AppendProtoOutput(const std::vector<std::vector<float>> & output)
    {
        auto & proto = (*_proto.mutable_outputs())[0];

        size_t offset = proto.mutable_data()->size();
        size_t nrows = output.size();
        size_t ncols = output[0].size();
        proto.mutable_data()->resize(proto.mutable_data()->size() + nrows*ncols * sizeof(float));

        char * p = &(*proto.mutable_data())[0] + offset;
        for (const auto & seq : output)
        {
            size_t size = seq.size() * sizeof(float);
            memcpy(p, &seq[0], size);
            p += size;
        }

        (*proto.mutable_shape())[0] += (int32_t)nrows;
        proto.set_count(proto.shape()[0] * proto.shape()[1]);
    }

    void Sequential::LoadModel()
    {
        if (_proto.model_uuid().size() > 0)
        {
            // Load from the cache
            auto it = gModelCache.find(_proto.model_uuid());
            if (it == gModelCache.end())
                throw runtime_error("The cached model [" + _proto.model_uuid() + "] was not found");
            _model = it->second;
            return;
        }

        auto jroot = json::parse(_proto.predict_params().c_str());
        bool cache = jroot.value<bool>("cache", true);

        if (_proto.model_path().size() > 0)
        {
            // Load from path
            auto path = utils::ToWide(_proto.model_path());
            _model = CNTK::Function::Load(path, globals::device);
            if (!cache) return;
            string uuid = utils::GenerateUuid();
            gModelCache[uuid] = _model;
            _proto.set_model_uuid(uuid);
            return;
        }

        if (_proto.model().size() > 0)
        {
            _model = cntk_utils::LoadModel(_proto.model().data(), _proto.model().size());
            if (!cache) return;
            string uuid = utils::GenerateUuid();
            gModelCache[uuid] = _model;
            _proto.set_model_uuid(uuid);
            return;
        }

        throw std::runtime_error("Bad model");
    }

    void Sequential::Predict()
    {
        LoadModel();

        // Setup the inputs and the outputs
        _inputVariables.clear();

        auto inputs = _model->Inputs();
        for (auto input : inputs)
        {
            if (!input.IsInput())
                continue;

            _inputVariables.push_back(input);
        }

        for (auto i = 0; i < _proto.inputs_size(); ++i)
            _bufferMinibatchSource->Add(_proto.inputs().Get(i), _inputVariables.at(i).Shape());
        _nsamples = _proto.inputs().Get(0).shape().Get(0);

        for (auto output : _model->Outputs())
            _inputVariables.push_back(output);

        CNTK::StreamInformation featureStreamInfo = _bufferMinibatchSource->FeatureStreamInfo();
        cntk_utils::DataBuffer labels(CNTK::DataType::Float, CNTK::NDShape({ _nsamples }).AppendShape(_inputVariables.back().Shape()));

        InitProtoOutput(globals::dataType, _nsamples, _inputVariables.back().Shape().TotalSize());

        vector<vector<float>> valueData;

        while (true)
        {
            auto minibatchData = _bufferMinibatchSource->GetNextMinibatch(_batchSize, globals::device);
            unordered_map<CNTK::Variable, CNTK::ValuePtr> outputMap = { { _inputVariables.back(), nullptr } };
            _model->Evaluate({ { _inputVariables[0], minibatchData.begin()->second.data } }, outputMap);

            outputMap.begin()->second.get()->CopyVariableValueTo(_inputVariables.back(), valueData);
            AppendProtoOutput(valueData);

            if (minibatchData.begin()->second.sweepEnd)
                break;
        }
    }
}
