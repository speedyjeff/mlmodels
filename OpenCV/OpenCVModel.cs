using System;
using System.Collections.Generic;
using Common;
using OpenCvSharp;
using OpenCvSharp.ML;

namespace OpenCV
{
    public class OpenCVModel : IModel<float[,], float[]>
    {
        public override void Train(float[,] data, List<float> labels)
        {
            if (TrainedModel != null) throw new InvalidOperationException("May only train/load a model once");
            if (data.GetLength(0) != labels.Count) throw new InvalidOperationException("Input data and label length must match");

            var dataInput = InputArray.Create<float>(data);
            var labelInput = InputArray.Create<float>(labels);

            TrainedModel = RTrees.Create();
            TrainedModel.Train(dataInput, SampleTypes.RowSample, labelInput);
        }

        public override void Load(string path)
        {
            if (TrainedModel != null) throw new InvalidOperationException("May only train/load a model once");

            TrainedModel = RTrees.Load(path);
        }

        public override void Save(string path)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before saving");

            TrainedModel.Save(path);
        }

        // return r^2
        public override double Evaluate(List<float[]> data, List<float> labels)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before evaluating");
            if (data.Count != labels.Count) throw new InvalidOperationException("Input data and label length must match");

            // R = (Count (sum of ab) - (sum of a)(sum of b)) / [sqrt((Count(sum a^2) - (sum of a)^2)(Count *(sum of b^2) - (sum of b)^2)]
            //  a == labels
            //  b == prediction
            var ab = 0d;
            var a = 0d;
            var b = 0d;
            var a2 = 0d;
            var b2 = 0d;
            for (int i = 0; i < data.Count; i++)
            {
                var pred = Predict(data[i]);

                a += labels[i];
                b += pred;
                ab += (labels[i] * pred);
                a2 += Math.Pow(labels[i], 2);
                b2 += Math.Pow(pred, 2);
            }
            var r2 = ((data.Count * ab) - (a * b)) / Math.Sqrt(((data.Count * a2) - Math.Pow(a, 2)) * (data.Count * b2 - Math.Pow(b, 2)));
            r2 = Math.Pow(r2, 2);

            return r2;
        }

        public override float Predict(float[] data)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before predicting");

            lock (TrainedModel)
            {
                using (var arr = InputArray.Create<float>(data))
                {
                    return TrainedModel.Predict(arr);
                }
            }
        }

        public override float[] Convert(float[] raw)
        {
            // trim the last column, as it is the label
            var data = new float[raw.Length - 1];
            for (int i = 0; i < data.Length; i++) data[i] = raw[i];
            return data;
        }

        public override float[,] Convert(List<float[]> raw)
        {
            if (raw == null || raw.Count == 0 || raw[0] == null || raw[0].Length == 0) throw new InvalidOperationException("Must provide a non zero set of input");

            // convert from List<float[]> to float[,]
            var data = new float[raw.Count, raw[0].Length];
            for(int i=0; i<raw.Count; i++)
            {
                for(int j=0; j<raw[i].Length; j++)
                {
                    data[i, j] = raw[i][j];
                }
            }

            return data;
        }

        #region private
        private RTrees TrainedModel;
        #endregion
    }
}
