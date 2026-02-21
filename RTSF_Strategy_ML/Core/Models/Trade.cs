using System;
using RTSF_Strategy_ML.Core.Enums;

namespace RTSF_Strategy_ML.Core.Models
{
    public class Trade
    {
        public int Id { get; set; }
        public TradeDirection Direction { get; set; }
        public DateTime EntryTime { get; set; }
        public DateTime ExitTime { get; set; }
        public float EntryPrice { get; set; }
        public float ExitPrice { get; set; }
        public int Contracts { get; set; }
        
        // PnL in points per 1 contract (before commission and multiplier)
        public float Pnl1c { get; set; }
        
        // Maximum Favorable Excursion and Maximum Adverse Excursion per 1 contract
        public float Mfe1c { get; set; }
        public float Mae1c { get; set; }
        
        // Net profit in currency (after slippage/commission and multiplier)
        public float PnlNet { get; set; }
        
        // Number of bars held
        public int BarsHeld { get; set; }
        
        // Reason for exit (signal_reverse, expiration, flip, end_of_day, etc)
        public string ExitReason { get; set; } = string.Empty;
        
        // ML Score (probability of success)
        public float MlScore { get; set; }

        public override string ToString()
        {
            return $"Trade {Id} [{Direction}] {EntryTime:yyyy-MM-dd HH:mm} -> {ExitTime:yyyy-MM-dd HH:mm} | Entry: {EntryPrice} Exit: {ExitPrice} | Contracts: {Contracts} | PnL 1c: {Pnl1c} Net: {PnlNet} | Reason: {ExitReason}";
        }
    }
}
