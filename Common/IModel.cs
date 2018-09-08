using System;
using System.Collections.Generic;

namespace Common
{
    public abstract class IModel<T, K>
    {
        public virtual void Train(T data, List<float> labels = null) { }
        public virtual void Load(string path) { }
        public virtual void Save(string path) { }
        // return r^2
        public virtual double Evaluate(List<K> data, List<float> labels = null)
        {
            return 0;
        }
        public virtual float Predict(K data)
        {
            return 0;
        }

        public virtual K Convert(float[] raw)
        {
            return default(K);
        }

        public virtual T Convert(List<K> raw)
        {
            return default(T);
        }
    }
}
