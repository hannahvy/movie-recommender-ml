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
            MLContext mlcontext = new MLContext(); // initializing creates new ML.NET enviornment
            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlcontext); // calls LoadData
            ITransformer model = BuildAndTrainModel(mlcontext, trainingDataView); // calls BuildAndTrainModel
            EvaluateModel(mlcontext, testDataView, model); // calls EvaluateModel
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

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            // define the data transformations
            // MapValueToKey() transformers userId and movieId to numeric key type Feature column
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            // choose machine learning algorithm and append to data transformation
            // my recommendatoin training algorithm is matrix factorization
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            // fit the model to train data and return trained model
            Console.WriteLine("=============== Training the model ===============");

            // Fit() trains model with provided training data set, executes Estimator and returns trained model, Transformer
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            // transform the test data
            Console.WriteLine("=============== Evaluating the model ===============");

            // Transform() makes predictions for multiple provided input rows of test dataset
            var prediction = model.Transform(testDataView);

            // evaluate the model
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            // print evaluation metrics to console
            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

    }
}
