#include "CntkUtils.h"

#include <iostream>
#include <fstream>

using namespace std;

namespace keras
{
    namespace cntk_utils
    {
        CNTK::FunctionPtr LoadModel(const char * buffer, size_t len)
        {
            auto tempPath = _wtmpnam(nullptr);
            ofstream outStream(tempPath, ofstream::trunc | ofstream::binary);
            outStream.write(buffer, len);
            outStream.close();
            auto result = CNTK::Function::Load(tempPath, globals::device);
            _wunlink(tempPath);

            return result;
        }
    }
}
