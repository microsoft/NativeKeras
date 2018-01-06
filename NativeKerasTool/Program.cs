using Keras;
using System;
using System.IO;
using TensorSharp;

namespace SharpKerasTool
{
    class Program
    {
        private static string GetDataPath(string relativePath)
        {
            string dir = Directory.GetCurrentDirectory();
            while(dir != null)
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
            // FitMnistQuick();
            FitMnist();
            // PredictMnist(GetDataPath("models/mnist.model"), GetDataPath("datasets/nda_mnist/mnist_xtest.nda"), GetDataPath("datasets/nda_mnist/mnist_ytest.nda"));
        }

        static void FitMnist()
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

        static void FitMnistQuick()
        {
            var model = new Sequential();
            model.Add(new Conv2D(32, kernelSize: new int[] { 3, 3 }, inputShape: new int[] { 28, 28, 1 }, activation: "relu"));
            model.Add(new Conv2D(64, kernelSize: new int[] { 3, 3 }, activation: "relu"));
            model.Add(new MaxPooling2D(poolSize: new int[] { 2, 2 }));
            model.Add(new Dropout(0.25));
            model.Add(new Flatten());
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(10, activation: "softmax"));

            var optimizer = new SGD(lr: 0.01);
            model.Compile("categorical_crossentropy", optimizer, new string[] { "accuracy" });

            var xtrain = TensorUtils.Deserialize(File.OpenRead("datasets/nda_mnist/mnist_xtrain.nda"));
            var ytrain = TensorUtils.Deserialize(File.OpenRead("datasets/nda_mnist/mnist_ytrain.nda"));

            xtrain = xtrain.Cast(DType.Float32);
            xtrain = Ops.Div(null, xtrain, 255f);

            ytrain = ytrain.Cast(DType.Float32);

            xtrain = xtrain.Narrow(0, 0, 2000);
            ytrain = ytrain.Narrow(0, 0, 2000);

            model.Fit(xtrain, ytrain, batchSize: 128, epochs: 3);

            var stream = new FileStream("c:/ttt/mnist.model", FileMode.OpenOrCreate, FileAccess.Write);
            stream.SetLength(0);

            model.Save(stream);
        }

        static void PredictMnist(string modelPath, string xtestPath, string ytestPath = null)
        {
            var xtest = TensorUtils.Deserialize(File.OpenRead(xtestPath));
            xtest = xtest.Cast(DType.Float32);
            xtest = Ops.Div(null, xtest, 255f);

            // xtest = xtest.Narrow(0, 0, 101);

            var model = Sequential.Load(modelPath);

            var result = model.Predict(xtest, batchSize: 32);

            if (ytestPath == null) return;

            var ytest = TensorUtils.Deserialize(File.OpenRead(ytestPath));
            ytest = ytest.Cast(DType.Float32);
            // ytest = ytest.Narrow(0, 0, 101);
            ytest = Ops.Argmax(null, ytest, 1).Squeeze();

            var t = result.Narrow(0, 0, 11);
            // Console.WriteLine(t.Format());

            result = Ops.Argmax(null, result, 1).Squeeze();

            t = result.Narrow(0, 0, 11);
            // Console.WriteLine(t.Format());

            double sum = 0.0;
            for (var i = 0; i < ytest.Sizes[0]; ++i)
                sum += (int)ytest.GetElementAsFloat(i) == (int)result.GetElementAsFloat(i) ? 1.0 : 0.0;

            Console.WriteLine($"Accuracy: {sum / ytest.Sizes[0] * 100}%");
        }
    }
}
