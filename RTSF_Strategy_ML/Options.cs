using System;
using CommandLine;

namespace RTSF_Strategy_ML
{
    public class Options
    {
        [Option('m', "mode", Required = true, HelpText = "Mode of operation: 'backtest' or 'live'.")]
        public string Mode { get; set; } = string.Empty;

        [Option('d', "data", Required = true, HelpText = "Path to the historical data folder (e.g., G:\\data2).")]
        public string DataPath { get; set; } = string.Empty;

        [Option("start", Required = false, Default = "2006-01-01", HelpText = "Start date for backtest (yyyy-MM-dd).")]
        public string StartDate { get; set; } = "2006-01-01";

        [Option("end", Required = false, Default = "2025-12-31", HelpText = "End date for backtest (yyyy-MM-dd).")]
        public string EndDate { get; set; } = "2025-12-31";

        [Option('t', "ticker", Required = false, Default = "RTSF", HelpText = "Ticker name.")]
        public string Ticker { get; set; } = "RTSF";

        [Option("signalr", Required = false, HelpText = "SignalR Hub URL (Required for 'live' mode).")]
        public string SignalRUrl { get; set; } = string.Empty;

        [Option("ml_model", Required = false, HelpText = "Path to the XGBoost ONNX model.", Default = "ML/block_e_xgboost_overlay.onnx")]
        public string MlModelPath { get; set; } = "ML/block_e_xgboost_overlay.onnx";

        [Option("ml_threshold", Required = false, Default = 0.5f, HelpText = "Threshold for ML filter probability.")]
        public float MlThreshold { get; set; } = 0.5f;

        [Option("leverage", Required = false, Default = 4.0f, HelpText = "Leverage multiplier.")]
        public float Leverage { get; set; } = 4.0f;
    }
}
