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
#include "Sequential.h"
#include "Utils.h"

#pragma warning (push)
#pragma warning (disable: 4251)
#pragma warning (disable: 4751)
#pragma warning (disable: 4800)
#include "KerasProto.pb.h"
#pragma warning (pop)

#include "DataBuffer.h"
#include "BufferMinibatchSource.h"

using namespace std;
using namespace nlohmann;
namespace cntk = CNTK;

extern "C" __declspec(dllexport) void KerasFitModel(
    const char * inData, unsigned inLen,
    char ** outData, unsigned * outLen,
    uint64_t * outPtr,
    char ** exceptionString, unsigned * exceptionLen,
    uint64_t * outExceptionPtr)
{
    try
    {
        keras::Sequential model;
        model.Init(inData, inLen);

        model.DriveCommand();

        model.GetOutput(outData, outLen);
        *outPtr = (uint64_t)*outData;

        *exceptionString = nullptr;
        *exceptionLen = 0;
        *outExceptionPtr = 0;
    }
    catch (const exception & e)
    {
        *outData = nullptr;
        *outLen = 0;
        *outPtr = 0;

        *exceptionLen = (unsigned)strlen(e.what());
        *exceptionString = new char[*exceptionLen + 1];
        strcpy(*exceptionString, e.what());
        *outExceptionPtr = (uint64_t)*exceptionString;
    }
}

extern "C" __declspec(dllexport) void KerasDeletePointer(void * ptr)
{
    delete[] ptr;
}