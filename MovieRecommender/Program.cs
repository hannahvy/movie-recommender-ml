using System;
using System.IO; // added manually
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML; // added manually
using Microsoft.ML.Trainers; // added manually
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace MovieRecommender
{
    public class Program
    {
        public static void Main(string[] args)
        {
            CreateHostBuilder(args).Build().Run(); // here by default
            MLContext mlcontext = new MLContext(); // initializing creates new ML.NET enviornment
            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlcontext); // calls LoadData
        }

        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            // initialize data path variables
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            // load data from csv files
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            // return train and test data
            return (trainingDataView, testDataView);
        }
        public static IHostBuilder CreateHostBuilder(string[] args) => // here by default
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                });
    }
}
