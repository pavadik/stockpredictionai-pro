using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace RTSF_Strategy_ML.ML
{
    public class XGBoostScorer : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName = string.Empty;

        public XGBoostScorer(string modelPath)
        {
            _session = new InferenceSession(modelPath);
            
            // Getting the input name of the model
            var inputMeta = _session.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                _inputName = name;
                break;
            }
        }

        /// <summary>
        /// Predict probability of successful trade given 14 float features.
        /// </summary>
        /// <param name="features">Float array of exactly 14 elements matching training</param>
        /// <returns>Probability between 0 and 1</returns>
        public float PredictProbability(float[] features)
        {
            if (features.Length != 14)
                throw new ArgumentException("Features array must have exactly 14 elements.");

            // Create tensor (batch_size=1, features=14)
            var inputTensor = new DenseTensor<float>(features, new int[] { 1, 14 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };

            using var results = _session.Run(inputs);
            
            // XGBoost ONNX typically outputs "probabilities" as a sequence of maps or a tensor.
            // For binary classification, the probability tensor often has shape [1, 2] where [0, 1] is the prob of class 1.
            // We need to extract the second output (usually named 'probabilities')
            foreach (var result in results)
            {
                if (result.Name == "probabilities" || result.Name == "output_probability") // names vary based on onnxmltools version
                {
                    // Handle output formats. 
                    // Sometimes it's SequenceOfMaps: IEnumerable<IDictionary<Int64, float>>
                    if (result.Value is IEnumerable<IDictionary<Int64, float>> maps)
                    {
                        foreach (var map in maps)
                        {
                            return map[1]; // class 1 probability
                        }
                    }
                    else if (result.Value is Tensor<float> probTensor)
                    {
                        return probTensor[0, 1]; // batch 0, class 1
                    }
                }
            }

            // Fallback for different ONNX export versions
            // Try to find any tensor output that looks like probabilities
            foreach (var result in results)
            {
                if (result.Value is Tensor<float> tensor && tensor.Length >= 2)
                {
                    return tensor[0, 1];
                }
            }

            throw new InvalidOperationException("Could not extract probability from ONNX model output.");
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
