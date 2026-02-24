using System;

namespace RTSF_Strategy_ML.Core.Models
{
    /// <summary>
    /// Represents a single aligned row of data for strategy calculation.
    /// Combines TF2 (trading timeframe) bars with the latest completed TF1 (trend timeframe) bars,
    /// along with pre-calculated filters and true range.
    /// </summary>
    public class StrategyDataRow
    {
        public DateTime Time { get; set; }
        
        // TF2 (Trading timeframe) data
        public float Open { get; set; }
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
        public long Volume { get; set; }
        
        // TF1 (Trend timeframe) data mapped via 'as-of' backward join
        public float CloseTf1 { get; set; }
        public float HighTf1 { get; set; }
        public float LowTf1 { get; set; }
        
        // Expiration filter: false if inside an expiration week block
        public bool AllowTrade { get; set; }
        
        // Intraday elapsed minutes since the first TF2 bar of the current trading day
        public int ElapsedMinutes { get; set; }
        
        // True if this is the last bar of the trading week
        public bool IsWeekEnd { get; set; }
        
        // Pre-calculated True Range (on TF2) for ATR calculation
        public float TrueRange { get; set; }

        // --- Extracted indicators ---
        public float TrendScore { get; set; }
        public float Tsa { get; set; }
        public float SmaTf1 { get; set; }
        
        public float TrendScore2 { get; set; }
        public float Tsa2 { get; set; }
        public float SmaTf2 { get; set; }

        public float Atr { get; set; }
        
        // --- Signal flags ---
        public bool InTimeWindow { get; set; }
        public bool EntrySignal { get; set; }
        public bool ExitSignal { get; set; }
        
        // --- Position Sizing ---
        public int ContractsBase { get; set; }
        public int Contracts { get; set; } // After leverage and caps
    }
}
