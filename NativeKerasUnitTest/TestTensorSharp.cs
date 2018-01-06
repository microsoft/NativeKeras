using Keras;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Cpu;

namespace SharpKerasUnitTest
{
    [TestClass]
    
    public class TestTensorSharp
    {
        [TestMethod, TestCategory("TensorSharp")]
        public void TestCopy()
        {
            var shape = new long[] { 60, 48, 64, 3 };
            var totalSize = 60 * 48 * 64 * 3;
            var data = Enumerable.Range(1, totalSize).Select(i => (float)i + 0.1f).ToArray();

            var allocator = new CpuAllocator();
            var t1 = new Tensor(allocator, DType.Float32, shape);
            t1.CopyFrom(data);
            var t2 = new Tensor(allocator, DType.Int32, shape);

            Ops.Copy(t2, t1);

            Assert.AreEqual(t2.ElementType, DType.Int32);

            Assert.AreEqual(t1.GetElementAsFloat(0, 0, 0, 0), 1.1f, float.Epsilon);
            Assert.AreEqual(t2.GetElementAsFloat(0, 0, 0, 0), 1f, float.Epsilon);
        }

        [TestMethod, TestCategory("TensorSharp")]
        public void TestDiv()
        {
            var shape = new long[] { 60, 48, 64, 3 };
            var totalSize = 60 * 48 * 64 * 3;
            var data = Enumerable.Range(1, totalSize).Select(i => (float)i + 0.1f).ToArray();

            var allocator = new CpuAllocator();
            var t1 = new Tensor(allocator, DType.Float32, shape);
            t1.CopyFrom(data);
            var t2 = Ops.Div(null, t1, 255f);

            for (var i = 0; i < shape[0]; ++i)
            {
                for (var j = 0; j < shape[1]; ++j)
                {
                    for (var k = 0; k < shape[2]; ++k)
                    {
                        for (var l = 0; l < shape[3]; ++l)
                        {
                            Assert.AreEqual(t1.GetElementAsFloat(i, j, k, l) / 255, t2.GetElementAsFloat(i, j, k, l), 0.0001);
                        }
                    }
                }
            }
        }

        [TestMethod, TestCategory("TensorSharp")]
        public void TestPermute()
        {
            // Matrix transpose
            var shape = new long[] { 45, 23 };
            var totalSize = shape.Aggregate((a, l) => a * l);
            var data = Enumerable.Range(1, (int)totalSize).Select(i => (float)i).ToArray();

            var t1 = TensorUtils.Create(shape, data);
            var t2 = TensorUtils.Create(shape, data);

            t2 = t2.Permute(1, 0);

            for (var i = 0; i < shape[0]; ++i)
                for (var j = 0; j < shape[1]; ++j)
                    Assert.AreEqual(t1.GetElementAsFloat(i, j), t2.GetElementAsFloat(j, i));

            Assert.AreEqual(t1.Sizes[0], shape[0]);
            Assert.AreEqual(t1.Sizes[1], shape[1]);

            Assert.AreEqual(t2.Sizes[0], shape[1]);
            Assert.AreEqual(t2.Sizes[1], shape[0]);

            // ND transpose
            shape = new long[] { 600, 28, 37, 3 };
            totalSize = shape.Aggregate((a, l) => a * l);
            data = Enumerable.Range(1, (int)totalSize).Select(i => (float)i).ToArray();

            t1 = TensorUtils.Create(shape, data);
            t2 = t1.View(new long[] { shape[0], 1, shape[1], shape[2], shape[3] });
            t2 = t2.Permute(new int[] { 4, 3, 2, 1, 0 });

            for (var i = 0; i < shape[0]; ++i)
            {
                for (var j = 0; j < shape[1]; ++j)
                {
                    for (var k = 0; k < shape[2]; ++k)
                    {
                        for (var l = 0; l < shape[3]; ++l)
                        {
                            Assert.AreEqual(t1.GetElementAsFloat(i, j, k, l), t2.GetElementAsFloat(l, k, j, 0, i));
                        }
                    }
                }
            }
        }

        [TestMethod, TestCategory("TensorSharp")]
        public void TestLoadMnist()
        {
            string path = "../../../datasets/nda_mnist/mnist_xtrain.nda";
            var nda = TensorUtils.Deserialize(File.OpenRead(path));
            CollectionAssert.AreEqual(nda.Sizes, new long[] { 60000, 28, 28 });
            Assert.AreEqual(nda.GetElementAsFloat(1, 0, 0), 0f);
            Assert.AreEqual(nda.GetElementAsFloat(0, 6, 9), 36f);

            // Cast the array to float
            nda = nda.Cast(DType.Float32);
            CollectionAssert.AreEqual(nda.Sizes, new long[] { 60000, 28, 28 });
            Assert.AreEqual(nda.GetElementAsFloat(1, 0, 0), 0f);
            Assert.AreEqual(nda.GetElementAsFloat(0, 6, 9), 36f);

            // Divide by 255
            nda = Ops.Div(null, nda, 255f);
            Assert.AreEqual(nda.GetElementAsFloat(1, 0, 0), 0f);
            Assert.AreEqual(nda.GetElementAsFloat(0, 6, 9), 36f / 255f, float.Epsilon);
            Assert.AreEqual(nda.GetElementAsFloat(0, 6, 9), 36.0 / 255, 0.000001);
        }
    }
}
