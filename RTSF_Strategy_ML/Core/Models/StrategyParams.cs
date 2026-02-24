using System;
using RTSF_Strategy_ML.Core.Enums;

namespace RTSF_Strategy_ML.Core.Models
{
    public class StrategyParams
    {
        // TF1 (Trend)
        public int Lookback { get; set; } = 165;
        public int Length { get; set; } = 180;
        public int Tf1Minutes { get; set; } = 210;
        public float Koeff1 { get; set; } = 1.0f;
        
        // TF2 (Trading)
        public int Lookback2 { get; set; } = 30;
        public int Length2 { get; set; } = 103;
        public int Tf2Minutes { get; set; } = 90;
        public float Koeff2 { get; set; } = 1.005f;

        // Intraday
        public int MinS { get; set; } = 290;
        public int MaxS { get; set; } = 480;

        // Position Sizing & Capital
        public int Mmcoff { get; set; } = 17;
        public float Capital { get; set; } = 5_000_000f;
        public float PointValueMult { get; set; } = 300f;

        // Caps & Leverage (from Combo Strategy)
        public int MaxContracts { get; set; } = 50;
        public float Leverage { get; set; } = 4.0f;

        // Trade Management
        public int ExitDay { get; set; } = 0;
        public int SdelDay { get; set; } = 0;
        public int ExitWeek { get; set; } = 0;

        // Stop Loss
        public string SlType { get; set; } = "none";
        public float SlPts { get; set; } = 0f;
        public float SlAtrMult { get; set; } = 0f;
        public float TrailAtr { get; set; } = 0f;
        public float TrailPts { get; set; } = 0f;
        public float BeTarget { get; set; } = 0f;
        public float BeOffset { get; set; } = 0f;
        public int TimeBars { get; set; } = 0;
        public float TimeTarget { get; set; } = 0f;

        // Target Direction
        public TradeDirection Direction { get; set; } = TradeDirection.Long;
    }
}
