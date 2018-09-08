using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet
{
    public class DataSet
    {
        [Column("0")]
        public float C0;
        [Column("1")]
        public float C1;
        [Column("2")]
        public float C2;
        [Column("3")]
        public float C3;
        [Column("4")]
        public float C4;
        [Column("5")]
        public float C5;
        [Column("6")]
        public float C6;
        [Column("7")]
        public float C7;
        [Column("8")]
        public float C8;
        [Column("9")]
        public float C9;
        [Column("10")]
        public float C10;
        [Column("11")]
        public float C11;
        [Column("12")]
        public float C12;
        [Column("13")]
        public float C13;
        [Column("14")]
        public float C14;
        [Column("15")]
        public float C15;
        [Column("16")]
        public float C16;
        [Column("17")]
        public float C17;
        [Column("18")]
        public float C18;
        [Column("19")]
        public float C19;
        [Column("20")]
        public float C20;
        [Column("21")]
        public float C21;
        [Column("22")]
        public float C22;
        [Column("23")]
        public float C23;
        [Column("24")]
        public float C24;
        [Column("25")]
        public float C25;

        // outcome
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
    }

    public class DataSetPrediction
    {
        [ColumnName("Score")]
        public float Score;
    }
}
