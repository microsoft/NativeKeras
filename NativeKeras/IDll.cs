using System;

namespace Keras
{
    interface IDll
    {
        Delegate GetDelegate<T>(string name);
    }
}
