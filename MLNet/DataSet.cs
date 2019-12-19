using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet
{
    public class DataSet
    {
        [LoadColumn(0)]
        public float C0;
        [LoadColumn(1)]
        public float C1;
        [LoadColumn(2)]
        public float C2;
        [LoadColumn(3)]
        public float C3;
        [LoadColumn(4)]
        public float C4;
        [LoadColumn(5)]
        public float C5;
        [LoadColumn(6)]
        public float C6;
        [LoadColumn(7)]
        public float C7;
        [LoadColumn(8)]
        public float C8;
        [LoadColumn(9)]
        public float C9;
        [LoadColumn(10)]
        public float C10;
        [LoadColumn(11)]
        public float C11;
        [LoadColumn(12)]
        public float C12;
        [LoadColumn(13)]
        public float C13;
        [LoadColumn(14)]
        public float C14;
        [LoadColumn(15)]
        public float C15;
        [LoadColumn(16)]
        public float C16;
        [LoadColumn(17)]
        public float C17;
        [LoadColumn(18)]
        public float C18;
        [LoadColumn(19)]
        public float C19;
        [LoadColumn(20)]
        public float C20;
        [LoadColumn(21)]
        public float C21;
        [LoadColumn(22)]
        public float C22;
        [LoadColumn(23)]
        public float C23;
        [LoadColumn(24)]
        public float C24;
        [LoadColumn(25)]
        public float C25;

        // outcome
        [LoadColumnName("Score")]
        public float Score;

        public static string[] ColumnNames()
        {
            return new string[]
            {
                "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
                "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
                "C20", "C21", "C22", "C23", "C24", "C25"
            };
        }

        public IEnumerable<float> ColumnValues()
        {
            yield return C0;
            yield return C1;
            yield return C2;
            yield return C3;
            yield return C4;
            yield return C5;
            yield return C6;
            yield return C7;
            yield return C8;
            yield return C9;
            yield return C10;
            yield return C11;
            yield return C12;
            yield return C13;
            yield return C14;
            yield return C15;
            yield return C16;
            yield return C17;
            yield return C18;
            yield return C19;
            yield return C20;
            yield return C21;
            yield return C22;
            yield return C23;
            yield return C24;
            yield return C25;
        }
    }

    public class DataSetPrediction
    {
        [LoadColumnName("Score")]
        public float Score;
    }
}
