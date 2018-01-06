using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Keras
{
    public interface IProgressWriter
    {
        void OnTrainingBegin(Dictionary<string, double> kvs);
        void OnTrainingEnd(Dictionary<string, double> kvs);

        void OnEpochBegin(uint epoch, Dictionary<string, double> kvs);
        void OnEpochEnd(uint epoch, Dictionary<string, double> kvs);

        void OnBatchBegin(uint batch, Dictionary<string, double> kvs);
        void OnBatchEnd(uint batch, Dictionary<string, double> kvs);
    }

    public class ProgressWriter : IProgressWriter
    {
        private uint _nepochs;
        private uint _nsamples;

        private uint _epochDigits;
        private uint _sampleDigits;

        private string _epochFormat;
        private string _batchFormat;

        private uint _epochSamples;

        private Dictionary<string, double> _accumulated;

        public ProgressWriter(uint nepochs, uint nsamples)
        {
            _nepochs = nepochs;
            _nsamples = nsamples;

            _epochDigits = (uint)Math.Floor(Math.Log10(_nepochs)) + 1;
            _sampleDigits = (uint)Math.Floor(Math.Log10(_nsamples)) + 1;

            _epochFormat = $"Epoch {{0,{_epochDigits}:d}}/{{1:d}}";
            _batchFormat = $"{{0,{_sampleDigits}:d}}/{{1:d}} [{{2}}] {{3,6:f}}% -- acc: {{4,6:f4}} -- loss: {{5,7:f4}}";

            _accumulated = new Dictionary<string, double>();
        }

        public void OnBatchBegin(uint batch, Dictionary<string, double> kvs)
        {
        }

        public void OnBatchEnd(uint batch, Dictionary<string, double> kvs)
        {
            var nsamples = kvs["nsamples"];
            if (_accumulated.Keys.Contains("nsamples"))
                _accumulated["nsamples"] += nsamples;
            else
                _accumulated["nsamples"] = nsamples;

            foreach (var kv in kvs)
            {
                if (kv.Key == "nsamples")
                    continue;

                if (_accumulated.Keys.Contains(kv.Key))
                    _accumulated[kv.Key] += nsamples * kv.Value;
                else
                    _accumulated[kv.Key] = nsamples * kv.Value;
            }

            _epochSamples += (uint)nsamples;

            var pct = (double)_epochSamples / _nsamples * 100;
            Console.Write("\r");
            Console.Write(string.Format(_batchFormat, _epochSamples, _nsamples, "", pct, _accumulated["acc"]/_accumulated["nsamples"], _accumulated["loss"]/_accumulated["nsamples"]));
        }

        public void OnEpochBegin(uint epoch, Dictionary<string, double> kvs)
        {
            _epochSamples = 0;

            Console.WriteLine();
            Console.WriteLine(string.Format(_epochFormat, epoch + 1, _nepochs));
        }

        public void OnEpochEnd(uint epoch, Dictionary<string, double> kvs)
        {
            Console.WriteLine();
        }

        public void OnTrainingBegin(Dictionary<string, double> kvs)
        {
        }

        public void OnTrainingEnd(Dictionary<string, double> kvs)
        {
        }
    }
}
