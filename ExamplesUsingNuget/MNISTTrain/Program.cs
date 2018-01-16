using System;
using Keras;
using System.IO;
using System.Linq;
using TensorSharp;

namespace MNISTTrain
{
    public class Program
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

        static void Main(string[] args)
        {
            // FitMnist();
            FitMnistSimple();
        }

        public static void FitMnist()
        {
            var model = new Sequential();
            model.Add(new Conv2D(32, kernelSize: new int[] { 3, 3 }, inputShape: new int[] { 28, 28, 1 }, activation: "relu"));
            model.Add(new Conv2D(64, kernelSize: new int[] { 3, 3 }, activation: "relu"));
            // model.Add(new MaxPooling1D(poolSize: 2));
            model.Add(new MaxPooling2D(poolSize: new int[] { 2, 2 }));
            model.Add(new Dropout(0.25));
            model.Add(new Flatten());
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(10, activation: "softmax"));

            var optimizer = new SGD(lr: 0.01);
            model.Compile("categorical_crossentropy", optimizer, new string[] { "accuracy" });

            var xtrain = TensorUtils.Deserialize(new FileStream(GetDataPath("datasets/nda_mnist/mnist_xtrain.nda"), FileMode.Open));
            var ytrain = TensorUtils.Deserialize(new FileStream(GetDataPath("datasets/nda_mnist/mnist_ytrain.nda"), FileMode.Open));

            xtrain = xtrain.Cast(DType.Float32);
            xtrain = Ops.Div(null, xtrain, 255f);

            ytrain = ytrain.Cast(DType.Float32);

            model.Fit(xtrain, ytrain, batchSize: 128, epochs: 12);

            var stream = new FileStream("c:/ttt/mnist.model", FileMode.OpenOrCreate, FileAccess.Write);
            stream.SetLength(0);

            model.Save(stream);
        }

        public static void FitMnistSimple()
        {
            var model = new Sequential();
            model.Add(new Dense(512, activation: "relu", inputShape: new int[] { 784 }));
            model.Add(new Dropout(0.2));
            model.Add(new Dense(512, activation: "relu"));
            model.Add(new Dropout(0.2));
            model.Add(new Dense(10, activation: "softmax"));

            var optimizer = new SGD(lr: 0.01);
            model.Compile("categorical_crossentropy", optimizer, new string[] { "accuracy" });

            var xtrain = TensorUtils.Deserialize(new FileStream(GetDataPath("datasets/nda_mnist/mnist_xtrain.nda"), FileMode.Open));
            var ytrain = TensorUtils.Deserialize(new FileStream(GetDataPath("datasets/nda_mnist/mnist_ytrain.nda"), FileMode.Open));

            xtrain = xtrain.Cast(DType.Float32);
            xtrain = Ops.Div(null, xtrain, 255f);

            ytrain = ytrain.Cast(DType.Float32);

            model.Fit(xtrain, ytrain, batchSize: 128, epochs: 20);

            var stream = new FileStream("c:/ttt/mnist-simple.model", FileMode.OpenOrCreate, FileAccess.Write);
            stream.SetLength(0);

            model.Save(stream);
        }
    }
}
