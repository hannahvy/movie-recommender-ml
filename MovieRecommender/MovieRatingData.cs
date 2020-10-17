using System;
using Microsoft.ML.Data;

namespace MovieRecommender
{
    public class MovieRating // specifies input data class
    {
        [LoadColumn(0)] // specifies which columns (by index) should be loaded
        public float userId; // feature
        [LoadColumn(1)]
        public float movieId; // feature
        [LoadColumn(2)]
        public float Label; // rating
    }

    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }
}