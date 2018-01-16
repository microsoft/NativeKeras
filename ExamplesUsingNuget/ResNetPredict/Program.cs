using Keras;
using OpenCvSharp;
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Cpu;

namespace ResNetPredict
{
    class Program
    {
        private static string GetDataPath(string relativePath)
        {
            string dir = Directory.GetCurrentDirectory();
            while (dir != null)
            {
                string path = Path.Combine(dir, relativePath);
                if (Directory.Exists(path) || File.Exists(path))
                    return path;
                dir = Path.GetDirectoryName(dir);
            }
            return null;
        }

        private static Tensor JpegToTensor(string path, int width, int height)
        {
            var mat = new Mat(path).Resize(new Size(height, height));

            var fmat = new Mat(new Size(height, width), MatType.CV_32FC3);
            mat.ConvertTo(fmat, MatType.CV_32FC3);

            var tensor = new Tensor(new CpuAllocator(), DType.Float32, new long[] { 1, fmat.Height, fmat.Width, fmat.Channels() });
            tensor.Storage.CopyToStorage(0, fmat.Data, fmat.Height * fmat.Width * fmat.Channels() * 4);
            return tensor;
        }

        static void Main(string[] args)
        {
            var modelPath = "models/resnet50.model";
            var width = 224;

            var tensor = JpegToTensor(GetDataPath("datasets/towers/et1.jpg"), width, width);

            var model = Sequential.Load(GetDataPath(modelPath));

            var prediction1 = model.Predict(tensor).Squeeze();

            tensor = JpegToTensor(GetDataPath("datasets/towers/et2.jpg"), width, width);

            var prediction2 = model.Predict(tensor).Squeeze();
            Console.WriteLine($" Distance to picture of the same object: {TensorUtils.EuclideanDistance(prediction1, prediction2).ToString("F2")}");

            tensor = JpegToTensor(GetDataPath("datasets/towers/et3.jpg"), width, width);

            prediction2 = model.Predict(tensor).Squeeze();
            Console.WriteLine($" Distance to picture of the same object: {TensorUtils.EuclideanDistance(prediction1, prediction2).ToString("F2")}");

            tensor = JpegToTensor(GetDataPath("datasets/towers/pt1.jpg"), width, width);

            prediction2 = model.Predict(tensor).Squeeze();
            Console.WriteLine($" Ddistance to picture of a different object: {TensorUtils.EuclideanDistance(prediction1, prediction2).ToString("F2")}");
        }
    }
}
