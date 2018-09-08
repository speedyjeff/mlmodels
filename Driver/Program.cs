using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

using Common;
using MLNet;
using OpenCV;

namespace Driver
{
    class Data<T>
    {
        public List<T> Train;
        public List<float> TrainLabels;
        public List<T> Eval;
        public List<float> EvalLabels;

        // timing information
        public double Training;
        public double Saving;
        public double SavingSize;
        public double Loading;
        public double Evaluating;
        public double Predicting;
        public double PPredicting;
        public int Predictions;

        // fitness
        public double RSquared;

        public Data()
        {
            Train = new List<T>();
            TrainLabels = new List<float>();
            Eval = new List<T>();
            EvalLabels = new List<float>();
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // params
            var timer = new Stopwatch();
            var mldata = new Data<DataSet>() { Predictions = 1000 };
            var cvdata = new Data<float[]>() { Predictions = 1000 };
            var ml = new MLNetModel();
            var cv = new OpenCVModel();

            Console.WriteLine("Reading data from disk...");
            // read in input from disk and create a training and evaluation set
            var count = 0;
            foreach(var line in File.ReadAllLines(@"Data\data.csv"))
            {
                // last column is the item to predict
                var row = line
                    .Trim()
                    .Split(',')
                    .Select(l => Convert.ToSingle(l))
                    .ToArray();
                count++;

                // use the first 50,000 as eval
                if (count < 50000)
                {
                    mldata.Eval.Add( ml.Convert(row) );
                    cvdata.Eval.Add( cv.Convert(row) );
                    cvdata.EvalLabels.Add(row[row.Length - 1]);
                }
                else
                {
                    mldata.Train.Add( ml.Convert(row) );
                    cvdata.Train.Add( cv.Convert(row) );
                    cvdata.TrainLabels.Add(row[row.Length - 1]);
                }
            }
            Console.WriteLine("Training ml({0}) cv({1}), Eval ml({2}) cv({3})", mldata.Train.Count, cvdata.Train.Count, mldata.Eval.Count, cvdata.Eval.Count);

            // training
            Console.WriteLine("ML Training...");
            timer.Reset();
            var mlinput = ml.Convert(mldata.Eval);
            timer.Start();
            ml.Train(mlinput);
            timer.Stop();
            mldata.Training = timer.ElapsedMilliseconds;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            Console.WriteLine("CV Training...");
            timer.Reset();
            var cvinput = cv.Convert(cvdata.Train);
            timer.Start();
            cv.Train(cvinput, cvdata.TrainLabels);
            timer.Stop();
            cvdata.Training = timer.ElapsedMilliseconds;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            // saving
            Console.WriteLine("ML Saving...");
            timer.Reset();
            timer.Start();
            ml.Save(@"ml.model");
            timer.Stop();
            mldata.Saving = timer.ElapsedMilliseconds;
            mldata.SavingSize = new System.IO.FileInfo(@"ml.model").Length;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            Console.WriteLine("CV Saving...");
            timer.Reset();
            timer.Start();
            cv.Save(@"cv.model");
            timer.Stop();
            cvdata.Saving = timer.ElapsedMilliseconds;
            cvdata.SavingSize = new System.IO.FileInfo(@"cv.model").Length;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            // loading
            Console.WriteLine("ML Loading...");
            ml = new MLNetModel();
            timer.Reset();
            timer.Start();
            ml.Load(@"ml.model");
            timer.Stop();
            mldata.Loading = timer.ElapsedMilliseconds;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            Console.WriteLine("CV Loading...");
            cv = new OpenCVModel();
            timer.Reset();
            timer.Start();
            cv.Load(@"cv.model");
            timer.Stop();
            cvdata.Loading = timer.ElapsedMilliseconds;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            // evaluating
            Console.WriteLine("ML Evauluating...");
            mlinput = ml.Convert(mldata.Eval);
            timer.Reset();
            timer.Start();
            var r2 = ml.Evaluate(mlinput);
            timer.Stop();
            mldata.Evaluating = timer.ElapsedMilliseconds;
            mldata.RSquared = r2;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            Console.WriteLine("CV Evauluating...");
            timer.Reset();
            timer.Start();
            r2 = cv.Evaluate(cvdata.Eval, cvdata.EvalLabels);
            timer.Stop();
            cvdata.Evaluating = timer.ElapsedMilliseconds;
            cvdata.RSquared = r2;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            // predicting
            Console.WriteLine("ML Predict...");
            timer.Reset();
            timer.Start();
            for(int i=0; i< mldata.Predictions; i++)
            {
                ml.Predict(mldata.Eval[i]);
            }
            timer.Stop();
            mldata.Predicting = timer.ElapsedMilliseconds;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            Console.WriteLine("CV Predict...");
            timer.Reset();
            timer.Start();
            for (int i = 0; i < cvdata.Predictions; i++)
            {
                cv.Predict(cvdata.Eval[i]);
            }
            timer.Stop();
            cvdata.Predicting = timer.ElapsedMilliseconds;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            // parallel predicting
            Console.WriteLine("ML Parallel Predict...");
            mlinput = mldata.Eval.Take(mldata.Predictions).ToList();
            timer.Reset();
            timer.Start();
            Parallel.ForEach(mlinput, (input) =>
            {
                ml.Predict(input);
            });
            timer.Stop();
            mldata.PPredicting = timer.ElapsedMilliseconds;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            Console.WriteLine("CV Parallel Predict...");
            var cvinput2 = cvdata.Eval.Take(cvdata.Predictions);
            timer.Reset();
            timer.Start();
            Parallel.ForEach(cvinput2, (input) =>
            {
                cv.Predict(input);
            });
            timer.Stop();
            cvdata.PPredicting = timer.ElapsedMilliseconds;
            Console.WriteLine("... {0}ms", timer.ElapsedMilliseconds);

            // results
            Console.WriteLine();
            Console.WriteLine(" \t| ML\t\t| CV\t\t| ML/CV");
            Console.WriteLine("Train\t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.Training, cvdata.Training, mldata.Training / cvdata.Training);
            Console.WriteLine("Save\t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.Saving, cvdata.Saving, mldata.Saving / cvdata.Saving);
            Console.WriteLine("Size\t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.SavingSize, cvdata.SavingSize, mldata.SavingSize / cvdata.SavingSize);
            Console.WriteLine("Load\t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.Loading, cvdata.Loading, mldata.Loading / cvdata.Loading);
            Console.WriteLine("Eval\t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.Evaluating, cvdata.Evaluating, mldata.Evaluating / cvdata.Evaluating);
            Console.WriteLine("R^2 \t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.RSquared, cvdata.RSquared, mldata.RSquared / cvdata.RSquared);
            Console.WriteLine("Pred\t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.Predicting, cvdata.Predicting, mldata.Predicting / cvdata.Predicting);
            Console.WriteLine("Pred/It\t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.Predicting/(float)mldata.Predictions, cvdata.Predicting / (float)cvdata.Predictions, (mldata.Predicting / (float)mldata.Predictions) / (cvdata.Predicting / (float)cvdata.Predictions));
            Console.WriteLine("Para P\t| {0,10:f2}\t| {1,10:f2}\t| {2,10:f2}", mldata.PPredicting, cvdata.PPredicting, mldata.PPredicting / cvdata.PPredicting);
        }

    }
}
