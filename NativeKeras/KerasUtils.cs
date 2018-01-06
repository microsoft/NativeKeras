using System;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.IO;
using Google.Protobuf;

namespace Keras
{
    public class KerasException : Exception
    {
        public KerasException(string message)
            : base(message)
        { }
    }

    public static class KerasUtils
    {
        public static int[] GetArray(object obj, int dimensions)
        {
            var type = obj.GetType();
            if (type == typeof(int))
                return Enumerable.Repeat((int)obj, dimensions).ToArray();
            else if (type == typeof(long))
                return Enumerable.Repeat((int)(long)obj, dimensions).ToArray();
            else if (type == typeof(ulong))
                return Enumerable.Repeat((int)(ulong)obj, dimensions).ToArray();
            else if (type == typeof(uint))
                return Enumerable.Repeat((int)(uint)obj, dimensions).ToArray();
            else if (type.IsArray)
            {
                var elementType = type.GetElementType();
                if (elementType == typeof(int))
                    return (int[])obj;
                else if (elementType == typeof(long))
                    return Array.ConvertAll((long[])obj, arg => (int)arg);
                else if (elementType == typeof(ulong))
                    return Array.ConvertAll((ulong[])obj, arg => (int)arg);
                else if (elementType == typeof(uint))
                    return Array.ConvertAll((uint[])obj, arg => (int)arg);
            }
            return null;
        }

        public static void AddActivation(JObject jobj, object activation)
        {
            AddActivation(jobj, "activation", activation);
        }

        public static void AddActivation(JObject jobj, string name, object activation)
        {
            if (activation != null)
            {
                if (activation.GetType() == typeof(string))
                    jobj[name] = (string)activation;
                else
                    throw new NotImplementedException("Only strings can be used for activation.");
            }
        }

        public static void AddInitializer(JObject jobj, string name, object initializer)
        {
            if (initializer == null)
                return;

            if (initializer.GetType() == typeof(string))
                jobj[name] = (string)initializer;
            else
                jobj[name] = (initializer as GraphOp).ToJObject();
        }

        public static void AddRegularizer(JObject jobj, string name, object regularizer)
        {
            if (regularizer == null)
                return;

            if (regularizer.GetType() == typeof(string))
                jobj[name] = (string)regularizer;
            else
                jobj[name] = (regularizer as GraphOp).ToJObject();
        }

        public static void AddStringOrObject(JObject jobj, string name, object obj)
        {
            if (obj == null)
                return;

            if (obj.GetType() == typeof(string))
                jobj[name] = (string)obj;
            else
                jobj[name] = (obj as GraphOp).ToJObject();
        }

        public static Dictionary<string, double> Values(this HistoryProto proto)
        {
            var result = new Dictionary<string, double>();
            for (var i = 0; i < proto.Values.Count; ++i)
                result.Add(proto.Names[i], proto.Values[i]);
            return result;
        }
    }
}
