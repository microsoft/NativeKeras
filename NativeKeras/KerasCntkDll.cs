using System;
using System.Runtime.InteropServices;

namespace Keras
{
    public class KerasCntkDll : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void KerasFitModelDelegate(byte[] inData, uint inlen, ref IntPtr outData, ref uint outLen, ref ulong outPtr, ref IntPtr exceptionData, ref uint exceptionLen, ref ulong exceptionPtr);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void KerasDeletePointerDelegate(ulong outPtr);

        private bool _disposed = false;

        private readonly IDll _dll;

        private KerasFitModelDelegate _fitModel;
        private KerasDeletePointerDelegate _deletePointer;

        private static KerasCntkDll _instance { get; } = new KerasCntkDll();

        private KerasCntkDll()
        {
            _dll = new WindowsDll("KerasCntk.dll");

            _fitModel = (KerasFitModelDelegate)_dll.GetDelegate<KerasFitModelDelegate>("KerasFitModel");
            _deletePointer = (KerasDeletePointerDelegate)_dll.GetDelegate<KerasDeletePointerDelegate>("KerasDeletePointer");
        }

        public static void KerasFitModel(byte[] inData, uint inlen, ref IntPtr outData, ref uint outLen, ref ulong outPtr, ref IntPtr exceptionData, ref uint exceptionLen, ref ulong exceptionPtr)
        {
            _instance._fitModel(inData, inlen, ref outData, ref outLen, ref outPtr, ref exceptionData, ref exceptionLen, ref exceptionPtr);
        }

        public static void KerasDeletePointer(ulong outPtr)
        {
            _instance._deletePointer(outPtr);
        }

        public void Dispose()
        {
            // Dispose of unmanaged resources.
            Dispose(true);
            // Suppress finalization.
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (disposing)
            {
                (_dll as IDisposable).Dispose();
            }

            // Free any unmanaged objects here.

            _disposed = true;
        }
    }
}
