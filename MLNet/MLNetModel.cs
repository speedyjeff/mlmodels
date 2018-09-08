using System;
using System.Collections.Generic;
using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLNet
{
    public class MLNetModel : IModel<List<DataSet>, DataSet>
    {
        public override void Train(List<DataSet> data, List<float> labels = null)
        {
            if (TrainedModel != null) throw new InvalidOperationException("May only train/load a model once");

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
        }

        public override void Load(string path)
        {
            if (TrainedModel != null) throw new InvalidOperationException("May only train/load a model once");

            TrainedModel = PredictionModel.ReadAsync<DataSet, DataSetPrediction>(path).Result;
        }

        public override void Save(string path)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before saving");

            TrainedModel.WriteAsync(path).Wait();
        }

        // return r^2
        public override double Evaluate(List<DataSet> data, List<float> labels = null)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before evaluating");

            lock (TrainedModel)
            {
                var testData = CollectionDataSource.Create(data);
                var evaluator = new RegressionEvaluator();
                var metrics = evaluator.Evaluate(TrainedModel, testData);

                return metrics.RSquared;
            }
        }

        public override float Predict(DataSet data)
        {
            if (TrainedModel == null) throw new InvalidOperationException("Must train/load a model before evaluating");

            lock (TrainedModel)
            {
                var result = TrainedModel.Predict(data);
                return result.Score;
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
                C25 = raw[25]
            };
        }

        public override List<DataSet> Convert(List<DataSet> raw)
        {
            // no conversion
            return raw;
        }

        #region private
        private PredictionModel<DataSet, DataSetPrediction> TrainedModel;
        #endregion
    }
}
