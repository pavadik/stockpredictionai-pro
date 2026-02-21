using System;
using System.IO;
using RTSF_Strategy_ML.ML;
using Xunit;

namespace RTSF_Strategy_ML.Tests
{
    public class XGBoostScorerTests
    {
        [Fact]
        public void PredictProbability_ValidModel_ReturnsProbability()
        {
            // Note: This test requires the ONNX model to be present in the ML folder
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "RTSF_Strategy_ML", "ML", "block_e_xgboost_overlay.onnx");
            modelPath = Path.GetFullPath(modelPath);

            if (!File.Exists(modelPath))
            {
                // Skip test if model is not available
                return;
            }

            using var scorer = new XGBoostScorer(modelPath);

            // Dummy features
            float[] features = new float[] { 
                10f, 0f, 0f, 1f, // hour, min, dow, month
                10000f, 500f, 25f, 0.01f, 0.02f, // d1_volume, d1_tr, d1_adx, d1_ret_1d, d1_ret_5d
                500f, 0.001f, -0.002f, 0.005f, // m1_volume, m1_ret_15m, m1_ret_60m, m1_ret_120m
                1f // is_long
            };

            float prob = scorer.PredictProbability(features);

            Assert.True(prob >= 0f && prob <= 1f);
        }

        [Fact]
        public void PredictProbability_InvalidFeaturesLength_ThrowsException()
        {
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "RTSF_Strategy_ML", "ML", "block_e_xgboost_overlay.onnx");
            modelPath = Path.GetFullPath(modelPath);

            if (!File.Exists(modelPath))
            {
                return;
            }

            using var scorer = new XGBoostScorer(modelPath);
            float[] features = new float[10]; // should be 14

            Assert.Throws<ArgumentException>(() => scorer.PredictProbability(features));
        }
    }
}
