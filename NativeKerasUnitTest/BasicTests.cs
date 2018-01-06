using Microsoft.VisualStudio.TestTools.UnitTesting;
using Keras;
using Newtonsoft.Json;

namespace SharpKerasUnitTest
{
    [TestClass]
    public class GraphTests
    {
        [TestMethod]
        public void TestBasic()
        {
            var model = new Sequential();
            model.Add(new Dense(64, inputShape: new int[] { 20 }, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(10, activation: "softmax"));
            var sgd = new SGD(lr: 0.01, momentum: 0.9, nesterov: true);
            model.Compile("categorical_crossentropy", sgd, new string[] { "accuracy" });
            var json = model.ToString(Formatting.Indented);
        }

        [TestMethod]
        public void TestFit()
        {
            var model = new Sequential();
            // model.Add(new Dense(64, inputShape: new Shape(784), activation: "relu"));
            model.Add(new Dense(64, inputShape: new int[] { 784 }));
            // model.Add(new Dense(128));
            var sgd = new SGD(lr: 0.003125);
            model.Compile("categorical_crossentropy", sgd, new string[] { "accuracy" });
        }

        [TestMethod]
        public void TestToJson()
        {
            var mp = new MaxPooling1D(poolSize: 2, padding: "same");
            var jobj = mp.ToJObject();

            var dropout = new Dropout(0.3, seed: 3984);
            jobj = dropout.ToJObject();
        }
    }
}
