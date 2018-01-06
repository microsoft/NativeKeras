__NativeKeras__ is an open source library for deep learning. It democratizes deep learning across languages via an easy to understand, and to use, language-agnostic API.
Bottom line is: no need of Python unless that's the language of your choice!

The API  is easy to use - it is inspired and closely follows
[Keras' API](https://keras.io). NativeKeras is implemented directly on top of [Microsoft's Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/).

NativeKeras can be used directly from C++. Furthermore, the library interface is designed around JSON and [Protocol Buffers](https://developers.google.com/protocol-buffers/).
Thus, it's extremely easy to use the API from any language which can talk JSON and Protocol Buffers!

As a sample implementatoin, NativeKeras provides a C# library.

# Installation
We are trying to figure out the build process. For now the nuget package is available [here](https://ivannp.visualstudio.com/nativekeras/_git/NativeKeras?path=%2FNativeKeras%2FNativeKeras.CpuOnly.0.0.1.nupkg&version=GBmaster&_a=contents).
To install the nuget, download it locally and then, from the [Package Manager Consoler](https://docs.microsoft.com/en-us/nuget/tools/package-manager-console) run:
    
    Install-Package C:\Path\To\The\Nuget\NativeKeras.CpuOnly.0.0.1.nupkg

# Status, Goals and Roadmap
The current code base implements a good number of basic Keras functionality like initializers and activations. The basic Keras layers (*Dense*, *Dropout*, etc)
are also implemented. Last, a good number of the image related layers is implemented as well.

Currently it is most useful to provide feedback via Pull Requests and Issues. The Issues help prioritize the development. What follows is a somewhat random
list of what is to come in near future (the plan is to release most of it before Christmas 2017):
* Provide a nuget repository and package for C# users
* Add more optimizers. Currently SGD is pretty much the only one supported and tested.
* Add Keras recurrent layers - *LSTM*
* Add Keras embedding layers
* Streamline CPU/GPU training/scoring process using an approach similar to Keras' global options

For the embedding and recurrent layers, the plan is to use [Keras IMDB example](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py).
If you have better ideas - please chime in.

# Setup to build NativeKeras
The project depends on a few C++ libraries. One way to add most dependencies is via [vcpkg](https://github.com/Microsoft/vcpkg):

	git clone https://github.com/Microsoft/vcpkg.git
	.\bootstrap-vcpkg.bat
	.\vcpkg integrate install
    .\vcpkg install protobuf:x64-windows
	.\vcpkg install nlohmann-json:x64-windows
	.\vcpkg install fmt:x64-windows
    .\vcpkg install boost:x64-windows

There are two dependencies which need to be installed by hand: __CNTK__ and __Torch's Tensor library__.

Both are relatively easy to build (for C++ libraries that is):
* CNTK: follow [the development instructions](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-development-environment). The Release_CpuOnly target is integrated in the project - see the instructions below.
* Torch's Tensor library: a self-contained, cmake project. Build only the TH library not the entire Torch project.

To hook them up, without changing any project settings:
* Torch's Tensor library: Copy the result folder of __MAKE_INSTALL__ under the nativekeras solution, renaming it to __th7__.
* CNTK: Create a folder called __cntk__ under the solution. From the CNTK repository, copy __CNTK/Source/CNTKv2LibraryDll/API__ folder to the newly created __cntk__ folder under the solution.
Now you have __cntk/API__. Similary, from the CNTK repository copy __CNTK/x64/Release_CpuOnly__, to __cntk/Release_CpuOnly__.

After these two steps, I would expect the project to build.

# Training MNIST, Keras-style
[Keras' MNIST example](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) coded in __nativekeras__:

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
    var xtrain = TensorUtils.Deserialize(new FileStream("datasets/nda_mnist/mnist_xtrain.nda", FileMode.Open));
    var ytrain = TensorUtils.Deserialize(new FileStream("datasets/nda_mnist/mnist_ytrain.nda", FileMode.Open));
    
    // The data is in row-major byte format. Convert to float and normalize.
    xtrain = xtrain.Cast(DType.Float32);
    xtrain = Ops.Div(null, xtrain, 255f);
    
    ytrain = ytrain.Cast(DType.Float32);
    
    // Kick off the training
    model.Fit(xtrain, ytrain, batchSize: 128, epochs: 12);

Reference [Keras Documentation](https://keras.io) directly for the meaning of each function and parameter used in the
deep network configuratoin.

Then to predict:

    // Load the test data
    var xtest = TensorUtils.Deserialize(File.OpenRead("datasets/nda_mnist/mnist_xtest.nda"));
    xtest = xtest.Cast(DType.Float32);
    xtest = Ops.Div(null, xtest, 255f);
    
    var model = Sequential.Load("models/mnist.model");
    
    // Predict
    var result = model.Predict(xtest, batchSize: 32);

All examples are available [from a separate repository](https://ivannp.visualstudio.com/NativeKerasExamples). Some are available 
[in this repository's Example folder](https://ivannp.visualstudio.com/_git/NativeKeras?path=%2FExamples).


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