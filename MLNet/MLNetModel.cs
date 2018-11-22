using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Common;
using Microsoft.ML;


#if ML_LEGACY
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
#else
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
#endif



namespace MLNet
{
    public class MLNetModel : IModel<List<DataSet>, DataSet>
    {
        public MLNetModel()
        {
            Context = new MLContext();
        }

        public override void Train(List<DataSet> data, List<float> labels = null)
        {
            if (TrainedModel != null) throw new InvalidOperationException("May only train/load a model once");

#if ML_LEGACY
             var pipeline = new LearningPipeline();

            // add data
            pipeline.Add(CollectionDataSource.Create(data));

            // choose what to predict
            pipeline.Add(new ColumnCopier(("Score", "Label")));
                    
            // add columns as features
            // do not include the features which should be predicted
            pipeline.Add(new ColumnConcatenator("Features", DataSet.ColumnNames()));

            // add a regression prediction
            pipeline.Add(new FastTreeRegressor());

            // train the model
            TrainedModel = pipeline.Train<DataSet, DataSetPrediction>();
#else
            // add data
            var textLoader = GetTextLoader(Context);

            // spill to disk !?!?! since there is no way to load from a collection
            var pathToData = "";
            try
            {
                // write data to disk
                pathToData = WriteToDisk(data);

                // read in data
                IDataView dataView = textLoader.Read(pathToData);

                // configurations
                var dataPipeline = Context.Transforms.CopyColumns("Score", "Label")
                    .Append(Context.Transforms.Concatenate("Features", DataSet.ColumnNames()));

                // set the training algorithm
                var trainer = Context.Regression.Trainers.FastTree(label: "Label", features: "Features");
                var trainingPipeline = dataPipeline.Append(trainer);

                TrainedModel = trainingPipeline.Fit(dataView);
            }
            finally
            {
                // cleanup
                if (!string.IsNullOrWhiteSpace(pathToData) && File.Exists(pathToData)) File.Delete(pathToData);
            }
#endif
        }

        public override void Load(string path)
        {
            if (TrainedModel != null) throw new InvalidOperationException("May only train/load a model once");

#if ML_LEGACY
            TrainedModel = PredictionModel.ReadAsync<DataSet, DataSetPrediction>(path).Result;
#else
            // load
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                TrainedModel = Context.Model.Load(stream);
            }

            // create the prediction function
            PredictFunc = TrainedModel.MakePredictionFunction<DataSet, DataSetPrediction>(Context);
#endif
        }

        public override void Save(string path)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before saving");

#if ML_LEGACY
            TrainedModel.WriteAsync(path).Wait();
#else
            // save
            using (var stream = File.Create(path))
            {
                TrainedModel.SaveTo(Context, stream);
            }
#endif
        }

        // return r^2
        public override double Evaluate(List<DataSet> data, List<float> labels = null)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before evaluating");

            lock (TrainedModel)
            {
#if ML_LEGACY
                var testData = CollectionDataSource.Create(data);
                //var evaluator = new RegressionEvaluator();
                //var metrics = evaluator.Evaluate(TrainedModel, testData);
                return 0;
                //return metrics.RSquared;
#else
                var textLoader = GetTextLoader(Context);

                var pathToData = "";
                try
                {
                    // ugh have to spill data to disk for it to work!
                    pathToData = WriteToDisk(data);

                    IDataView dataView = textLoader.Read(pathToData);
                    var predictions = TrainedModel.Transform(dataView);
                    var metrics = Context.Regression.Evaluate(predictions, label: "Label", score: "Score");

                    return metrics.RSquared;
                }
                finally
                {
                    // cleanup
                    if (!string.IsNullOrWhiteSpace(pathToData) && File.Exists(pathToData)) File.Delete(pathToData);
                }
#endif
            }
        }

        public override float Predict(DataSet data)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before evaluating");

            lock (TrainedModel)
            {
#if ML_LEGACY
                var result = TrainedModel.Predict(data);
                return result.Score;
#else
                var result = PredictFunc.Predict(data);
                return result.Score;
#endif
            }
        }

        public override DataSet Convert(float[] raw)
        {
            return new DataSet()
            {
                C0 = raw[0],
                C1 = raw[1],
                C2 = raw[2],
                C3 = raw[3],
                C4 = raw[4],
                C5 = raw[5],
                C6 = raw[6],
                C7 = raw[7],
                C8 = raw[8],
                C9 = raw[9],
                C10 = raw[10],
                C11 = raw[11],
                C12 = raw[12],
                C13 = raw[13],
                C14 = raw[14],
                C15 = raw[15],
                C16 = raw[16],
                C17 = raw[17],
                C18 = raw[18],
                C19 = raw[19],
                C20 = raw[20],
                C21 = raw[21],
                C22 = raw[22],
                C23 = raw[23],
                C24 = raw[24],
                C25 = raw[25],

                Score = raw[26]
            };
        }

        public override List<DataSet> Convert(List<DataSet> raw)
        {
            // no conversion
            return raw;
        }

#region private

#if ML_LEGACY
        private PredictionModel<DataSet, DataSetPrediction> TrainedModel;
#else
        private MLContext Context;
        private ITransformer TrainedModel;
        private PredictionFunction<DataSet, DataSetPrediction> PredictFunc;

        private static string WriteToDisk(List<DataSet> data)
        {
            // geneate a random path
            var path = Path.Combine(Path.GetTempPath(), Path.GetTempFileName());

            using (var writer = File.CreateText(path))
            {
                foreach (var d in data)
                {
                    foreach (var v in d.ColumnValues())
                    {
                        writer.Write(v);
                        writer.Write(',');
                    }
                    writer.WriteLine(d.Score);
                }
            }

            return path;
        }

        private static TextLoader GetTextLoader(MLContext context)
        {
            var index = 0;
            var columns = DataSet.ColumnNames().Select(c => new TextLoader.Column(c, DataKind.R4, index++)).ToList();
            columns.Add(new TextLoader.Column("Score", DataKind.R4, index));
            return context.Data.TextReader(
                new TextLoader.Arguments()
                {
                    Separator = ",",
                    HasHeader = false,
                    Column = columns.ToArray()
                });
        }
#endif

#endregion
    }
}
