using Google.Protobuf;
using Keras;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;

namespace Keras
{
    public static class TensorUtils
    {
        private static readonly Dictionary<DataType, DType> _dataTypeToDType = new Dictionary<DataType, DType>()
        {
            { DataType.Float, DType.Float32 },
            { DataType.Double, DType.Float64 },
            { DataType.Uint8, DType.UInt8 },
            { DataType.Int32, DType.Int32 }
        };

        private static readonly Dictionary<DType, DataType> _dtypeToDataType = new Dictionary<DType, DataType>()
        {
            { DType.Float32, DataType.Float },
            { DType.Float64, DataType.Double },
            { DType.UInt8, DataType.Uint8 },
            { DType.Int32, DataType.Int32 }
        };

        public static Tensor Create(long [] shape, float [] data)
        {
            var tensor = new Tensor(new CpuAllocator(), DType.Float32, shape);
            tensor.CopyFrom(data);
            return tensor;
        }

        public static Tensor Create(long[] shape, double[] data)
        {
            var tensor = new Tensor(new CpuAllocator(), DType.Float64, shape);
            tensor.CopyFrom(data);
            return tensor;
        }

        public static Tensor Create(long[] shape, int[] data)
        {
            var tensor = new Tensor(new CpuAllocator(), DType.Int32, shape);
            tensor.CopyFrom(data);
            return tensor;
        }

        public static Tensor Create(long[] shape, byte[] data)
        {
            var tensor = new Tensor(new CpuAllocator(), DType.UInt8, shape);
            tensor.CopyFrom(data);
            return tensor;
        }

        public static unsafe Tensor Deserialize(Stream stream)
        {
            var proto = TensorProto.Parser.ParseFrom(stream);
            return Deserialize(proto);
        }

        public static unsafe Tensor Deserialize(TensorProto proto)
        {
            if (!_dataTypeToDType.Keys.Contains(proto.Type))
                throw new Exception($"Tensors don't support '{proto.Type.ToString()}' data type");

            var allocator = new CpuAllocator();
            var dtype = _dataTypeToDType[proto.Type];
            var storage = (CpuStorage)allocator.Allocate(dtype, proto.Data.Length / dtype.Size());

            var bytes = proto.Data.ToByteArray();

            fixed (byte* p = bytes)
            {
                IntPtr ptr = (IntPtr)p;
                storage.CopyToStorage(0, ptr, bytes.Length);
            }

            var sizes = proto.Shape.Select(i => (long)i).ToArray();
            var strides = TensorDimensionHelpers.GetContiguousStride(sizes);
            return new Tensor(sizes, strides, storage, 0);
        }

        public static Tensor Cast(this Tensor tensor, DType dtype)
        {
            var result = new Tensor(tensor.Allocator, dtype, tensor.Sizes);
            Ops.Copy(result, tensor);
            return result;
        }

        public static unsafe TensorProto GetProto(this Tensor tensor)
        {
            var result = new TensorProto();
            var sizes = tensor.Sizes.Select(l => (int)l);
            result.Shape.Add(sizes);
            result.Count = sizes.Aggregate((a, i) => a * i);
            result.Type = _dtypeToDataType[tensor.ElementType];
            result.Format = TensorFormat.RowMajor;

            tensor = Ops.AsContiguous(tensor);
            var bytes = new byte[tensor.Storage.ByteLength];
            fixed (byte* p = bytes)
            {
                IntPtr ptr = (IntPtr)p;
                tensor.Storage.CopyFromStorage(ptr, 0, bytes.Length);
            }
            result.Data = ByteString.CopyFrom(bytes);
            return result;
        }

        public static bool Equals(Tensor t1, Tensor t2)
        {
            if (t1.ElementType != t2.ElementType) return false;
            if (!t1.Sizes.SequenceEqual(t2.Sizes)) return false;
            var t = Ops.EqualTo(null, t1, t2);
            t = t.View(new long[] { t.ElementCount() });
            for (var i = 0; i < t.ElementCount(); ++i)
                if (t.GetElementAsFloat(i) != 1.0) return false;
            return true;
        }

        public static double EuclideanDistance(Tensor t1, Tensor t2)
        {
            var diff = Ops.Pow(null, Ops.Sub(null, t1, t2), 2.0f);
            var t = diff.View(diff.ElementCount());
            double result = 0.0;
            for (var i = 0; i < t.ElementCount(); ++i)
                result += t.GetElementAsFloat(i);
            return Math.Sqrt(result);
        }
    }
}
