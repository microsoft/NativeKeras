using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Keras
{
    public class WindowsDll : IDll
    {
        private IntPtr _handle;
        private bool _disposed = false;

        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string path);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetProcAddress(IntPtr handle, string name);

        [DllImport("kernel32.dll")]
        private static extern bool FreeLibrary(IntPtr handle);

        public WindowsDll(string path)
        {
            _handle = LoadLibrary(path);
            if (_handle == IntPtr.Zero)
                throw new FileNotFoundException($"Failed to load {path}");
        }

        ~WindowsDll()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(false);
        }

        public Delegate GetDelegate<T>(string name)
        {
            var result = GetProcAddress(_handle, name);
            if (result == IntPtr.Zero)
                throw new Exception($"Error loading dll '{name}' functionality.");
            return Marshal.GetDelegateForFunctionPointer(result, typeof(T));
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
                // Free any other managed objects here.
            }

            // Free any unmanaged objects here.

            if (_handle != IntPtr.Zero)
            {
                FreeLibrary(_handle);
                _handle = IntPtr.Zero;
            }

            _disposed = true;
        }
    }
}
