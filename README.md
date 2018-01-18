__NativeKeras__ is an open source library for deep learning. NativeKeras is a high-level library,
vaguely based on [Keras](https://keras.io/) and [Gluon](https://github.com/gluon-api/gluon-api). At the
same time, NativeKeras is build in native code (C++). Thus, it's language agnostic in the sense that
it could be easily integrated into any language that speaks JSON and Protocol Buffers! Bottom line is:
no need of Python unless that's the language of your choice!

NativeKeras is implemented directly on top of [Microsoft's Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/).

As a sample implementatoin, NativeKeras provides a C# library.

# Status
The current code base implements a good number of basic Keras functionality like initializers
and activations. The basic Keras layers (*Dense*, *Dropout*, etc) are also implemented. Last,
a good number of the image related layers (Conv2D, MaxPooling2D, etc) is implemented as well.

# Limitations
NativeKeras is a *proof of concept* at this stage.
* Windows is the only supported platform
* No GPU use yet

# Installation

# Goals and Roadmap
NativeKeras C++ layers are similar to the ongoing work to integrate [ONNX](https://github.com/onnx/onnx)
into [CNTK](https://github.com/Microsoft/CNTK/). To avoid duplication, we are planning to add
new layers as they are added to CNTK. This is why NativeKeras still lacks, among others, **LSTM** and **Embedding** layers.

Our current priorities, in no specific order, are:
* Streamline the build process.
* Provide nuggets.
* Streamline CPU/GPU training/scoring process using an approach similar to Keras' global options.
* Add more optimizers. Currently SGD is pretty much the only one supported and tested.
* Move to .NET Core and support Linux.
* Extend the JSON model to support arbitrary (non-sequential) models.


# Setup to build NativeKeras
The project depends on a few C++ libraries. One way to add most dependencies is via [vcpkg](https://github.com/Microsoft/vcpkg):

	git clone https://github.com/Microsoft/vcpkg.git
	.\bootstrap-vcpkg.bat
	.\vcpkg integrate install
    .\vcpkg install protobuf:x64-windows
	.\vcpkg install nlohmann-json:x64-windows
	.\vcpkg install fmt:x64-windows
    .\vcpkg install boost:x64-windows
    .\vcpkg install gtest:x64-windows

There are two dependencies which need to be installed by hand: __CNTK__ and __Torch's Tensor library__.

Both are relatively easy to build (for C++ libraries that is):
* CNTK: follow [the development instructions](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-development-environment). The Release_CpuOnly target is integrated in the project - see the instructions below.
* Torch's Tensor library: a self-contained, cmake project. Build only the TH library not the entire Torch project.

To hook them up, without changing any project settings:
* Torch's Tensor library: Copy the result folder of __MAKE_INSTALL__ under the nativekeras solution, renaming it to __th7__.
* CNTK: Create a folder called __cntk__ under the solution. From the CNTK repository, copy __CNTK/Source/CNTKv2LibraryDll/API__ folder to the newly created __cntk__ folder under the solution.
Now you have __cntk/API__. Similary, from the CNTK repository copy __CNTK/x64/Release_CpuOnly__, to __cntk/Release_CpuOnly__.

The CNTK version changes rapidly, I usually have to update the CNTK library name in the Visual Studio linker input section.

After these two steps, I would expect the project to build.


# Training MNIST, Keras-style
[Keras' MNIST example](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) coded in NativeKeras:

    // Define the DNN
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
    
    // Load the training data. The code uses TensorSharp.
    var xtrain = TensorUtils.Deserialize(new FileStream("Datasets/nda_mnist/mnist_xtrain.nda", FileMode.Open));
    var ytrain = TensorUtils.Deserialize(new FileStream("Datasets/nda_mnist/mnist_ytrain.nda", FileMode.Open));
    
    // The data is in row-major byte format. Convert to float and normalize.
    xtrain = xtrain.Cast(DType.Float32);
    xtrain = Ops.Div(null, xtrain, 255f);
    
    ytrain = ytrain.Cast(DType.Float32);
    
    // Kick off the training
    model.Fit(xtrain, ytrain, batchSize: 128, epochs: 12);

Reference [Keras Documentation](https://keras.io) directly for the meaning of each function and parameter used in the
deep network configuratoin. Yes, it is that easy!

Then to predict:

    // Load the test data
    var xtest = TensorUtils.Deserialize(File.OpenRead("Datasets/nda_mnist/mnist_xtest.nda"));
    xtest = xtest.Cast(DType.Float32);
    xtest = Ops.Div(null, xtest, 255f);
    
    var model = Sequential.Load("Models/mnist.model");
    
    // Predict
    var result = model.Predict(xtest, batchSize: 32);

The examples are from [this repository's Example folder](https://github.com/Microsoft/NativeKeras/tree/master/Examples).


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
