using Google.Protobuf;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;

namespace Keras
{
    public static class Globals
    {
        public static string DataFormat { get; set; } = "channels_last";
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class GraphOp
    {
        [JsonProperty(PropertyName = "op", Required = Required.Always)]
        protected string _op;

        public virtual JObject ToJObject()
        {
            _op = _op ?? GetType().Name;
            var jobj = JObject.FromObject(this);
            return jobj;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class Activation : GraphOp
    {
        [JsonProperty(PropertyName = "activation")]
        private string _activation;

        public Activation(string activation)
        {
            _activation = activation;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class Dense : GraphOp
    {
        [JsonProperty(PropertyName = "activation")]
        private object _activation;
        [JsonProperty(PropertyName = "units")]
        private ulong _units;
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _inputShape;
        [JsonProperty(PropertyName = "kernel_initializer")]
        private object _kernelInitializer;

        public Dense(ulong units, object activation = null, int [] inputShape = null, object kernelInitializer = null)
        {
            _activation = activation;
            _units = units;
            _inputShape = inputShape;
            _kernelInitializer = kernelInitializer ?? "glorot_uniform";
        }

        public override JObject ToJObject()
        {
            var jobj = new JObject();
            jobj["units"] = _units;

            if (_inputShape != null)
                jobj["input_shape"] = new JArray(_inputShape);

            KerasUtils.AddActivation(jobj, _activation);

            if (_kernelInitializer.GetType() == typeof(string))
                jobj["kernel_initializer"] = (string)_kernelInitializer;
            else
                jobj["kernel_initializer"] = (_kernelInitializer as GraphOp).ToJObject();

            jobj["op"] = "Dense";
            return jobj;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class Dropout : GraphOp
    {
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int [] _inputShape;

        [JsonProperty(PropertyName = "rate")]
        public double Rate { get; set; }

        [JsonProperty(PropertyName = "seed", NullValueHandling = NullValueHandling.Ignore)]
        public uint? Seed { get; set; } = null;

        public Dropout(double rate, uint? seed = null, int[] inputShape = null)
        {
            Rate = rate;
            _inputShape = inputShape;
            Seed = seed;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class Flatten : GraphOp
    {
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _inputShape;

        public Flatten(int [] inputShape = null)
        {
            _inputShape = inputShape;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class Conv1D : GraphOp
    {
        private int[] _inputShape;
        private int _filters;
        private int _kernelSize;
        private int _strides;
        private object _activation;
        private bool _useBias;

        public Conv1D(int filters, int kernelSize, int strides = 1, object activation = null, bool useBias = true, int[] inputShape = null)
        {
            _inputShape = inputShape;
            _kernelSize = kernelSize;
            _filters = filters;
            _activation = activation;
            _useBias = useBias;
            _strides = strides;

            _op = "Conv1D";
        }

        public override JObject ToJObject()
        {
            var jobj = new JObject();
            jobj["filters"] = _filters;
            jobj["kernel_size"] = _kernelSize;
            jobj["strides"] = _strides;
            jobj["use_bias"] = _useBias;

            if (_inputShape != null)
                jobj["input_shape"] = new JArray(_inputShape);

            KerasUtils.AddActivation(jobj, _activation);

            jobj["bias_initializer"] = "zeros";

            jobj.Add("op", "Conv1D");
            return jobj;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class Conv2D : GraphOp
    {
        private int[] _inputShape;
        private int _filters;
        private int[] _kernelSize;
        private int[] _strides;
        private object _activation;
        private bool _useBias;

        public Conv2D(int filters, object kernelSize, object strides = null, object activation = null, bool useBias = true, int [] inputShape = null)
        {
            _inputShape = inputShape;
            _kernelSize = KerasUtils.GetArray(kernelSize, 2);
            if (_kernelSize == null)
                throw new ArgumentException("The kernelSize parameter type is not supported.");
            _filters = filters;
            _activation = activation;
            _useBias = useBias;
            if (strides == null)
                _strides = new int[] { 1, 1 };
            else
                _strides = KerasUtils.GetArray(strides, 2);

            _op = "Conv2D";
        }

        public override JObject ToJObject()
        {
            var jobj = new JObject();
            jobj["filters"] = _filters;
            jobj["kernel_size"] = new JArray(_kernelSize);
            jobj["strides"] = new JArray(_strides);
            jobj["use_bias"] = _useBias;

            if (_inputShape != null)
                jobj["input_shape"] = new JArray(_inputShape);

            KerasUtils.AddActivation(jobj, _activation);

            jobj["bias_initializer"] = "zeros";

            jobj.Add("op", "Conv2D");
            return jobj;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class MaxPooling1D : GraphOp
    {
        [JsonProperty(PropertyName = "pool_size", NullValueHandling = NullValueHandling.Ignore)]
        private int _poolSize;
        [JsonProperty(PropertyName = "strides", NullValueHandling = NullValueHandling.Ignore)]
        private int _strides;
        [JsonProperty(PropertyName = "padding", NullValueHandling = NullValueHandling.Ignore)]
        private string _padding;
        [JsonProperty(PropertyName = "data_format", NullValueHandling = NullValueHandling.Ignore)]
        private string _dataFormat;
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _inputShape;

        public MaxPooling1D(int poolSize = 2, int? strides = null, string padding = "valid", string dataFormat = null, int [] inputShape = null)
        {
            _poolSize = poolSize;
            _strides = strides.HasValue ? strides.Value : poolSize;
            _padding = padding;
            _dataFormat = dataFormat == null ? Globals.DataFormat : dataFormat;
            _inputShape = inputShape;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class MaxPooling2D : GraphOp
    {
        [JsonProperty(PropertyName = "pool_size", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _poolSize;
        [JsonProperty(PropertyName = "strides", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _strides;
        [JsonProperty(PropertyName = "padding", NullValueHandling = NullValueHandling.Ignore)]
        private string _padding;
        [JsonProperty(PropertyName = "data_format", NullValueHandling = NullValueHandling.Ignore)]
        private string _dataFormat;
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _inputShape;

        public MaxPooling2D(object poolSize = null, object strides = null, string padding = "valid", string dataFormat = null, int [] inputShape = null)
        {
            _poolSize = poolSize == null ? new int[] { 2, 2 } : KerasUtils.GetArray(poolSize, 2);
            _strides = strides == null ? _poolSize : KerasUtils.GetArray(strides, 2);
            _padding = padding;
            _dataFormat = dataFormat == null ? Globals.DataFormat : dataFormat;
            _inputShape = inputShape;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class AveragePooling1D : GraphOp
    {
        [JsonProperty(PropertyName = "pool_size", NullValueHandling = NullValueHandling.Ignore)]
        private int _poolSize;
        [JsonProperty(PropertyName = "strides", NullValueHandling = NullValueHandling.Ignore)]
        private int _strides;
        [JsonProperty(PropertyName = "padding", NullValueHandling = NullValueHandling.Ignore)]
        private string _padding;
        [JsonProperty(PropertyName = "data_format", NullValueHandling = NullValueHandling.Ignore)]
        private string _dataFormat;
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _inputShape;

        public AveragePooling1D(int poolSize = 2, int? strides = null, string padding = "valid", string dataFormat = null, int[] inputShape = null)
        {
            _poolSize = poolSize;
            _strides = strides.HasValue ? strides.Value : poolSize;
            _padding = padding;
            _dataFormat = dataFormat == null ? Globals.DataFormat : dataFormat;
            _inputShape = inputShape;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class AveragePooling2D : GraphOp
    {
        [JsonProperty(PropertyName = "pool_size", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _poolSize;
        [JsonProperty(PropertyName = "strides", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _strides;
        [JsonProperty(PropertyName = "padding", NullValueHandling = NullValueHandling.Ignore)]
        private string _padding;
        [JsonProperty(PropertyName = "data_format", NullValueHandling = NullValueHandling.Ignore)]
        private string _dataFormat;
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _inputShape;

        public AveragePooling2D(object poolSize = null, object strides = null, string padding = "valid", string dataFormat = null, int[] inputShape = null)
        {
            _poolSize = poolSize == null ? new int[] { 2, 2 } : KerasUtils.GetArray(poolSize, 2);
            _strides = strides == null ? _poolSize : KerasUtils.GetArray(strides, 2);
            _padding = padding;
            _dataFormat = dataFormat == null ? Globals.DataFormat : dataFormat;
            _inputShape = inputShape;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class GlobalMaxPooling1D : GraphOp
    {
        public GlobalMaxPooling1D()
        {
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class Embedding : GraphOp
    {
        [JsonProperty(PropertyName = "input_dim")]
        private ulong _inputDim;
        [JsonProperty(PropertyName = "output_dim")]
        private ulong _outputDim;
        [JsonProperty(PropertyName = "embedding_initializer", NullValueHandling = NullValueHandling.Ignore)]
        private object _embeddingInitializer;
        [JsonProperty(PropertyName = "embedding_regularizer", NullValueHandling = NullValueHandling.Ignore)]
        private object _embeddingRegularizer;
        [JsonProperty(PropertyName = "embedding_constraint", NullValueHandling = NullValueHandling.Ignore)]
        private object _embeddingConstraint;
        [JsonProperty(PropertyName = "mask_zero")]
        private bool _maskZero;
        [JsonProperty(PropertyName = "input_length")]
        private ulong? _inputLength;
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _inputShape;

        public Embedding(ulong inputDim, ulong outputDim, object embeddingInitializer = null, object embeddingRegularizer = null, object embeddingConstraint = null, bool maskZero = false, ulong? inputLength = null, int[] inputShape = null)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
            _embeddingInitializer = embeddingInitializer ?? "uniform";
            _embeddingRegularizer = embeddingRegularizer;
            _embeddingConstraint = embeddingConstraint;
            _maskZero = maskZero;
            _inputLength = inputLength;
            _inputShape = inputShape;
        }

        public override JObject ToJObject()
        {
            var jobj = new JObject()
            {
                ["input_dim"] = _inputDim,
                ["output_dim"] = _outputDim,
                ["mask_zero"] = _maskZero
            };

            KerasUtils.AddStringOrObject(jobj, "embedding_initializer", _embeddingInitializer);
            KerasUtils.AddStringOrObject(jobj, "embedding_regularizer", _embeddingRegularizer);

            if (_inputLength.HasValue)
                jobj.Add("input_length", _inputLength.Value);

            if (_inputShape != null)
                jobj["input_shape"] = new JArray(_inputShape);

            jobj["op"] = "Embedding";
            return jobj;
        }
    }

    // LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
    //   recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
    //   recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
    //   recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
    [JsonObject(MemberSerialization.OptIn)]
    public class LSTM : GraphOp
    {
        [JsonProperty(PropertyName = "units")]
        private ulong _units;
        [JsonProperty(PropertyName = "activation", NullValueHandling = NullValueHandling.Ignore)]
        private object _activation;
        [JsonProperty(PropertyName = "recurrent_activation", NullValueHandling = NullValueHandling.Ignore)]
        private object _recurrentActivation;
        [JsonProperty(PropertyName = "input_shape", NullValueHandling = NullValueHandling.Ignore)]
        private int[] _inputShape;
        [JsonProperty(PropertyName = "use_bias", NullValueHandling = NullValueHandling.Ignore)]
        private bool _useBias;
        [JsonProperty(PropertyName = "kernel_initializer", NullValueHandling = NullValueHandling.Ignore)]
        private object _kernelInitializer;
        [JsonProperty(PropertyName = "recurrent_initializer", NullValueHandling = NullValueHandling.Ignore)]
        private object _recurrentInitializer;
        [JsonProperty(PropertyName = "bias_initializer", NullValueHandling = NullValueHandling.Ignore)]
        private object _biasInitializer;
        [JsonProperty(PropertyName = "unit_forget_bias", NullValueHandling = NullValueHandling.Ignore)]
        private bool _unitForgetBias;
        [JsonProperty(PropertyName = "kernel_regularizer", NullValueHandling = NullValueHandling.Ignore)]
        private object _kernelRegularizer;
        [JsonProperty(PropertyName = "recurrent_regularizer", NullValueHandling = NullValueHandling.Ignore)]
        private object _recurrentRegularizer;
        [JsonProperty(PropertyName = "bias_regularizer", NullValueHandling = NullValueHandling.Ignore)]
        private object _biasRegularizer;
        [JsonProperty(PropertyName = "activity_regularizer", NullValueHandling = NullValueHandling.Ignore)]
        private object _activityRegularizer;
        [JsonProperty(PropertyName = "dropout", NullValueHandling = NullValueHandling.Ignore)]
        private double _dropout;
        [JsonProperty(PropertyName = "recurrent_dropout", NullValueHandling = NullValueHandling.Ignore)]
        private double _recurrentDropout;

        public LSTM(ulong units, object activation = null, object recurrentActivation = null, bool useBias = true, object kernelInitializer = null, object recurrentInitializer = null, object biasInitializer = null, bool unitForgetBias = true, object kernelRegularizer = null, object recurrentRegularizer = null, object biasRegularizer = null, double dropout = 0.0, double recurrentDropout = 0.0, int[] inputShape = null)
        {
            _units = units;
            _activation = activation ?? "tanh";
            _recurrentActivation = recurrentActivation ?? "hard_sigmoid";
            _useBias = useBias;
            _kernelInitializer = kernelInitializer ?? "glorot_uniform";
            _recurrentInitializer = recurrentInitializer ?? "orthogonal";
            _biasInitializer = biasInitializer ?? "zeros";
            _unitForgetBias = unitForgetBias;
            _inputShape = inputShape;

            _dropout = dropout;
            _recurrentDropout = recurrentDropout;
        }

        public override JObject ToJObject()
        {
            var jobj = new JObject();
            jobj["units"] = _units;

            if (_inputShape != null)
                jobj["input_shape"] = new JArray(_inputShape);

            jobj["use_bias"] = _useBias;

            KerasUtils.AddActivation(jobj, _activation);
            KerasUtils.AddActivation(jobj, "recurrent_activation", _recurrentActivation);

            KerasUtils.AddStringOrObject(jobj, "kernel_initializer", _kernelInitializer);
            KerasUtils.AddStringOrObject(jobj, "recurrent_initializer", _recurrentInitializer);
            KerasUtils.AddStringOrObject(jobj, "bias_initializer", _biasInitializer);

            jobj["unit_forget_bias"] = _unitForgetBias;

            KerasUtils.AddStringOrObject(jobj, "kernel_regularizer", _kernelRegularizer);
            KerasUtils.AddStringOrObject(jobj, "recurrent_regularizer", _recurrentRegularizer);
            KerasUtils.AddStringOrObject(jobj, "bias_regularizer", _biasRegularizer);
            KerasUtils.AddStringOrObject(jobj, "activity_regularizer", _activityRegularizer);

            jobj["dropout"] = _dropout;
            jobj["recurrent_dropout"] = _recurrentDropout;

            jobj["op"] = "LSTM";
            return jobj;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class SGD : GraphOp
    {
        [JsonProperty(PropertyName = "lr", NullValueHandling = NullValueHandling.Ignore)]
        public double LearningRate { get; set; } = 0.01;
        [JsonProperty(PropertyName = "momentum", NullValueHandling = NullValueHandling.Ignore)]
        public double Momentum { get; set; } = 0;
        [JsonProperty(PropertyName = "decay", NullValueHandling = NullValueHandling.Ignore)]
        public double Decay { get; set; } = 0;
        [JsonProperty(PropertyName = "nestorov", NullValueHandling = NullValueHandling.Ignore)]
        public bool Nestorov { get; set; } = false;

        public SGD(double lr = 0.01, double momentum = 0.0, double decay = 0.0, bool nesterov = false)
        {
            LearningRate = lr;
            Momentum = momentum;
            Decay = decay;
            Nestorov = nesterov;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class Adadelta : GraphOp
    {
        [JsonProperty(PropertyName = "lr", NullValueHandling = NullValueHandling.Ignore)]
        public double LearningRate { get; set; } = 0.01;
        [JsonProperty(PropertyName = "rho", NullValueHandling = NullValueHandling.Ignore)]
        public double RHO { get; set; } = 0.95;
        [JsonProperty(PropertyName = "epsilon", NullValueHandling = NullValueHandling.Ignore)]
        public double Epsilon { get; set; } = 1e-08;
        [JsonProperty(PropertyName = "decay", NullValueHandling = NullValueHandling.Ignore)]
        public double Decay { get; set; } = 0;

        public Adadelta(double lr = 1.0, double rho = 0.0, double epsilon = 0.0, double decay = 0.0)
        {
            LearningRate = lr;
            RHO = rho;
            Epsilon = epsilon;
            Decay = decay;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class RandomNormal : GraphOp
    {
        [JsonProperty(PropertyName ="mean")]
        private double _mean = 0;

        [JsonProperty(PropertyName = "stddev")]
        private double _stddev = 0.05;

        [JsonProperty(PropertyName = "seed", NullValueHandling = NullValueHandling.Ignore)]
        private ulong? _seed;

        public RandomNormal(double stddev, ulong? seed = null)
        {
            _stddev = stddev;
            _seed = seed;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class TruncatedNormal : GraphOp
    {
        [JsonProperty(PropertyName = "mean")]
        private double _mean = 0;

        [JsonProperty(PropertyName = "stddev")]
        private double _stddev = 0.05;

        [JsonProperty(PropertyName = "seed", NullValueHandling = NullValueHandling.Ignore)]
        private ulong? _seed;

        public TruncatedNormal(double mean, double stddev = 0.05, ulong? seed = null)
        {
            throw new KerasException("CNTK hard-codes mean to 0.0.");
        }

        public TruncatedNormal(double stddev = 0.05, ulong? seed = null)
        {
            _stddev = stddev;
            _seed = seed;
        }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class RandomUniform: GraphOp
    {
        [JsonProperty(PropertyName = "minval")]
        private double _minval = -0.05;

        [JsonProperty(PropertyName = "maxval")]
        private double _maxval = 0.05;

        [JsonProperty(PropertyName = "seed", NullValueHandling = NullValueHandling.Ignore)]
        private ulong? _seed;

        public RandomUniform(double minval = -0.05, double maxval = 0.05, ulong? seed = null)
        {
            _minval = minval;
            _maxval = maxval;
            _seed = seed;

            if (-_minval != _maxval)
                throw new KerasException("CNTK supports only symetric around zero uniform distribution [-minval must be equal to maxval].");
        }
    }

    public class Sequential
    {
        private JObject _graph;

        private byte[] _model;
        private string _uuid = "";
        private string _path = "";

        public Sequential()
        {
            _graph = new JObject();
            _graph["graph"] = new JArray();
        }

        public void Add<T>(T cmd) where T : GraphOp
        {
            var jarray = (JArray)_graph["graph"];
            jarray.Add(cmd.ToJObject());
        }

        public void Add(string jcmd)
        {
            var jarray = (JArray)_graph["graph"];
            jarray.Add(JObject.Parse(jcmd));
        }

        private JToken ObjectAsJValue(object obj)
        {
            var type = obj.GetType();
            if (typeof(string) == type)
            {
                return new JValue((string)obj);
            }
            else if (type.IsArray)
            {
                var jarray = new JArray();
                foreach (var element in (object[])obj)
                {
                    if (typeof(string) == element.GetType())
                    {
                        jarray.Add(new JValue((string)element));
                    }
                    else
                    {
                        jarray.Add(((GraphOp)element).ToJObject());
                    }
                }
                return jarray;
            }
            else
            {
                var cmd = (GraphOp)obj;
                return cmd.ToJObject();
            }
        }

        public void Compile(object loss, object optimizer, object metrics)
        {
            var jobj = new JObject();
            jobj["optimizer"] = ObjectAsJValue(optimizer);
            jobj["metrics"] = ObjectAsJValue(metrics);
            jobj["loss"] = ObjectAsJValue(loss);
            _graph["compile_params"] = jobj;
        }

        [DllImport(@"KerasCntk.dll")]
        public static extern void KerasFitModel(byte[] inData, uint inlen, ref IntPtr outData, ref uint outLen, ref ulong outPtr, ref IntPtr exceptionData, ref uint exceptionLen, ref ulong exceptionPtr);

        [DllImport(@"KerasCntk.dll")]
        public static extern void KerasDeletePointer(ulong outPtr);

        private static ProgressCallbackState _state;
        private static ProgressCallback _callback;

        public void Fit(string path, uint nsamples, uint nfeatures, uint nlabels, uint batchSize, uint epochs, uint verbose = 1)
        {
            KerasProto kerasProto = new KerasProto();
            kerasProto.Graph = _graph.ToString(Formatting.None);
            kerasProto.Path = path;
            kerasProto.Nsamples = nsamples;
            kerasProto.Nfeatures = nfeatures;
            kerasProto.Nlabels = nlabels;

            kerasProto.BatchSize = batchSize;
            kerasProto.Epochs = epochs;

            kerasProto.Verbose = verbose;

            using (var stream = new MemoryStream())
            {
                kerasProto.WriteTo(stream);
                var bytes = stream.ToArray();

                IntPtr outData = IntPtr.Zero;
                uint outLen = 0;
                ulong outPtr = 0;

                IntPtr exceptionData = IntPtr.Zero;
                uint exceptionLen = 0;
                ulong exceptionPtr = 0;

                KerasFitModel(bytes, (uint)bytes.Length, ref outData, ref outLen, ref outPtr, ref exceptionData, ref exceptionLen, ref exceptionPtr);
                // KerasCntkDll.KerasFitModel(bytes, (uint)bytes.Length, ref outData, ref outLen, ref outPtr, ref exceptionData, ref exceptionLen, ref exceptionPtr);

                if (exceptionLen == 0)
                {
                    _model = new byte[outLen];
                    Marshal.Copy(outData, _model, 0, (int)outLen);

                    KerasDeletePointer(outPtr);
                }
                else
                {
                    var outBytes = new byte[exceptionLen];
                    Marshal.Copy(exceptionData, outBytes, 0, (int)exceptionLen);

                    var exception = new KerasException(Encoding.ASCII.GetString(outBytes));
                    KerasDeletePointer(exceptionPtr);

                    throw exception;
                }
            }
        }

        public void Fit(Tensor x, Tensor y, uint batchSize = 32, uint epochs = 10, uint verbose = 1)
        {
            KerasProto kerasProto = new KerasProto();
            kerasProto.Graph = _graph.ToString(Formatting.None);

            kerasProto.BatchSize = batchSize;
            kerasProto.Epochs = epochs;

            kerasProto.Verbose = verbose;

            kerasProto.Inputs.Add(x.GetProto());
            kerasProto.Inputs.Add(y.GetProto());

            kerasProto.Command = KerasCommand.Fit;

            _state = new ProgressCallbackState(new ProgressWriter(epochs, (uint)x.Sizes[0]));
            _callback = new ProgressCallback(_state.Callback);

            kerasProto.ProgressCallback = (ulong)Marshal.GetFunctionPointerForDelegate(_callback);

            using (var stream = new MemoryStream())
            {
                kerasProto.WriteTo(stream);
                var bytes = stream.ToArray();

                IntPtr outData = IntPtr.Zero;
                uint outLen = 0;
                ulong outPtr = 0;

                IntPtr exceptionData = IntPtr.Zero;
                uint exceptionLen = 0;
                ulong exceptionPtr = 0;

                KerasFitModel(bytes, (uint)bytes.Length, ref outData, ref outLen, ref outPtr, ref exceptionData, ref exceptionLen, ref exceptionPtr);

                if (exceptionLen == 0)
                {
                    var resultBytes = new byte[outLen];
                    Marshal.Copy(outData, resultBytes, 0, (int)outLen);
                    KerasDeletePointer(outPtr);

                    var resultProto = KerasProto.Parser.ParseFrom(resultBytes);
                    _model = resultProto.Model.ToArray();
                }
                else
                {
                    var outBytes = new byte[exceptionLen];
                    Marshal.Copy(exceptionData, outBytes, 0, (int)exceptionLen);

                    var exception = new KerasException(Encoding.ASCII.GetString(outBytes));
                    KerasDeletePointer(exceptionPtr);

                    throw exception;
                }
            }
        }

        public Tensor Predict(Tensor x, uint batchSize = 32, uint verbose = 1, bool cache = true)
        {
            KerasProto kerasProto = new KerasProto();

            kerasProto.BatchSize = batchSize;
            kerasProto.Verbose = verbose;

            var jobj = new JObject()
            {
                ["cache"] = cache
            };
            kerasProto.PredictParams = jobj.ToString(Formatting.None);

            // TODO Consider not copying the model if we have a uuid
            if(_model != null) kerasProto.Model = ByteString.CopyFrom(_model);
            kerasProto.ModelUuid = _uuid;
            kerasProto.ModelPath = _path;
            kerasProto.Inputs.Add(x.GetProto());

            kerasProto.Command = KerasCommand.Predict;

            using (var stream = new MemoryStream())
            {
                kerasProto.WriteTo(stream);
                var bytes = stream.ToArray();

                IntPtr outData = IntPtr.Zero;
                uint outLen = 0;
                ulong outPtr = 0;

                IntPtr exceptionData = IntPtr.Zero;
                uint exceptionLen = 0;
                ulong exceptionPtr = 0;

                KerasFitModel(bytes, (uint)bytes.Length, ref outData, ref outLen, ref outPtr, ref exceptionData, ref exceptionLen, ref exceptionPtr);

                if (exceptionLen == 0)
                {
                    var resultBytes = new byte[outLen];
                    Marshal.Copy(outData, resultBytes, 0, (int)outLen);

                    KerasDeletePointer(outPtr);

                    var resultProto = KerasProto.Parser.ParseFrom(resultBytes);
                    _uuid = resultProto.ModelUuid;
                    return TensorUtils.Deserialize(resultProto.Outputs[0]);
                }
                else
                {
                    var outBytes = new byte[exceptionLen];
                    Marshal.Copy(exceptionData, outBytes, 0, (int)exceptionLen);

                    var exception = new KerasException(Encoding.ASCII.GetString(outBytes));
                    KerasDeletePointer(exceptionPtr);

                    throw exception;
                }
            }
        }

        public void Save(Stream stream)
        {
            using (var writer = new BinaryWriter(stream))
                writer.Write(_model, 0, _model.Length);
        }

        public override string ToString()
        {
            return _graph.ToString(Formatting.None);
        }

        public string ToString(Formatting formatting = Formatting.None)
        {
            return _graph.ToString(formatting);
        }

        public static Sequential Load(string path)
        {
            var result = new Sequential();
            result._path = path;
            return result;
        }

        public static Sequential Load(Stream stream)
        {
            var result = new Sequential();
            using (var buffer = new MemoryStream())
            {
                stream.CopyTo(buffer);
                result._model = buffer.ToArray();
            }
            return result;
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate void ProgressCallback(IntPtr data, uint length);

        private class ProgressCallbackState
        {
            private IProgressWriter _writer;

            public ProgressCallbackState(IProgressWriter writer)
            {
                _writer = writer;
            }

            public void Callback(IntPtr data, uint length)
            {
                var bytes = new byte[length];
                Marshal.Copy(data, bytes, 0, (int)length);

                var proto = HistoryProto.Parser.ParseFrom(bytes);
                var kvs = proto.Values();
                switch (proto.Type)
                {
                    case HistoryCallbackType.BatchBegin:
                        _writer.OnBatchBegin(proto.Id, kvs);
                        break;

                    case HistoryCallbackType.BatchEnd:
                        _writer.OnBatchEnd(proto.Id, kvs);
                        break;

                    case HistoryCallbackType.EpochBegin:
                        _writer.OnEpochBegin(proto.Id, kvs);
                        break;

                    case HistoryCallbackType.EpochEnd:
                        _writer.OnEpochEnd(proto.Id, kvs);
                        break;

                    case HistoryCallbackType.TrainingEnd:
                        _writer.OnTrainingEnd(kvs);
                        break;

                    case HistoryCallbackType.TrainingBegin:
                        _writer.OnTrainingBegin(kvs);
                        break;
                }
            }
        }
    }
}