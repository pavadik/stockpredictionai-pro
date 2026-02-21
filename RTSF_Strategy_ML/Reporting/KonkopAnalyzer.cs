using System;
using System.Collections.Generic;
using System.Linq;
using RTSF_Strategy_ML.Core.Models;

namespace RTSF_Strategy_ML.Reporting
{
    public class KonkopMetrics
    {
        public string Label { get; set; } = string.Empty;
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public float StartEquity { get; set; }
        public float EndEquity { get; set; }
        
        public int TotalTrades { get; set; }
        public int WinTrades { get; set; }
        public int LossTrades { get; set; }
        
        public float NetProfit { get; set; }
        public float GrossProfit { get; set; }
        public float GrossLoss { get; set; }
        
        public float ProfitFactor { get; set; }
        public float WinRate { get; set; }
        
        public float AvgTrade { get; set; }
        public float AvgWin { get; set; }
        public float AvgLoss { get; set; }
        
        public float LargestWin { get; set; }
        public float LargestLoss { get; set; }
        
        public int MaxConseqWin { get; set; }
        public int MaxConseqLoss { get; set; }
        
        public float MaxDdAbs { get; set; }
        public float MaxDdPett { get; set; }
        
        public float TotalReturnPct { get; set; }
        public float AnnualReturnPct { get; set; }
        
        public float MathExpectationPct { get; set; }
        public float AvgHoldHours { get; set; }
        public float Years { get; set; }
    }

    public class YearlyMetrics
    {
        public int Year { get; set; }
        public int Trades { get; set; }
        public float NetProfit { get; set; }
        public float ProfitFactor { get; set; }
        public float WinRate { get; set; }
        public float AvgTrade { get; set; }
        public float MaxDdPct { get; set; }
    }

    public static class KonkopAnalyzer
    {
        public static KonkopMetrics Analyze(List<Trade> trades, string label, float capital)
        {
            var m = new KonkopMetrics { Label = label, StartEquity = capital };
            
            if (trades == null || trades.Count == 0)
                return m;

            m.StartDate = trades.First().EntryTime;
            m.EndDate = trades.Last().ExitTime;
            
            TimeSpan duration = m.EndDate - m.StartDate;
            m.Years = (float)Math.Max(duration.TotalDays / 365.25, 0.01);

            m.TotalTrades = trades.Count;
            
            float equity = capital;
            float peakEquity = capital;

            int currentWins = 0;
            int currentLosses = 0;

            long totalHoldSeconds = 0;

            foreach (var t in trades)
            {
                float pnl = t.PnlNet;
                equity += pnl;

                if (equity > peakEquity) peakEquity = equity;
                
                float ddAbs = peakEquity - equity;
                float ddPct = ddAbs / peakEquity * 100f;

                if (ddAbs > m.MaxDdAbs) m.MaxDdAbs = ddAbs;
                if (ddPct > m.MaxDdPett) m.MaxDdPett = ddPct;

                if (pnl > 0)
                {
                    m.WinTrades++;
                    m.GrossProfit += pnl;
                    if (pnl > m.LargestWin) m.LargestWin = pnl;
                    
                    currentWins++;
                    currentLosses = 0;
                    if (currentWins > m.MaxConseqWin) m.MaxConseqWin = currentWins;
                }
                else if (pnl < 0)
                {
                    m.LossTrades++;
                    m.GrossLoss += Math.Abs(pnl);
                    if (pnl < m.LargestLoss) m.LargestLoss = pnl;
                    
                    currentLosses++;
                    currentWins = 0;
                    if (currentLosses > m.MaxConseqLoss) m.MaxConseqLoss = currentLosses;
                }

                totalHoldSeconds += (long)(t.ExitTime - t.EntryTime).TotalSeconds;
            }

            m.EndEquity = equity;
            m.NetProfit = equity - capital;
            
            m.WinRate = m.TotalTrades > 0 ? (float)m.WinTrades / m.TotalTrades * 100f : 0f;
            m.ProfitFactor = m.GrossLoss > 0 ? m.GrossProfit / m.GrossLoss : float.PositiveInfinity;

            m.AvgTrade = m.TotalTrades > 0 ? m.NetProfit / m.TotalTrades : 0f;
            m.AvgWin = m.WinTrades > 0 ? m.GrossProfit / m.WinTrades : 0f;
            m.AvgLoss = m.LossTrades > 0 ? m.GrossLoss / m.LossTrades : 0f;

            m.MathExpectationPct = m.AvgTrade / capital * 100f;
            m.TotalReturnPct = m.NetProfit / capital * 100f;
            m.AnnualReturnPct = ((float)Math.Pow(1 + m.NetProfit / capital, 1.0 / m.Years) - 1) * 100f;

            m.AvgHoldHours = m.TotalTrades > 0 ? (float)totalHoldSeconds / m.TotalTrades / 3600f : 0f;

            return m;
        }

        public static void PrintReport(KonkopMetrics m)
        {
            string sep = new string('=', 62);
            Console.WriteLine(sep);
            Console.WriteLine($"  {m.Label}");
            Console.WriteLine(sep);
            Console.WriteLine();
            Console.WriteLine($"  Start date       {m.StartDate:dd.MM.yyyy,14}    End date      {m.EndDate:dd.MM.yyyy,14}");
            Console.WriteLine($"  Start equity   {m.StartEquity,14:N0}    End equity  {m.EndEquity,14:N0}");
            Console.WriteLine();
            Console.WriteLine("  ─── Trade Statistics ─────────────────────────────────────");
            Console.WriteLine($"  Total trades          {m.TotalTrades,8}    Profit factor          {m.ProfitFactor,8:F2}");
            Console.WriteLine($"  Math. expectation  {m.MathExpectationPct,8:F2} %    Win rate               {m.WinRate,8:F2} %");
            Console.WriteLine();
            Console.WriteLine("  ─── Profit / Loss ───────────────────────────────────────");
            Console.WriteLine($"  Net profit       {m.NetProfit,14:N0}    Max DD             {m.MaxDdPett,8:F2} %");
            Console.WriteLine($"  Gross profit     {m.GrossProfit,14:N0}    Gross loss     {m.GrossLoss,14:N0}");
            Console.WriteLine();
            Console.WriteLine("  ─── Win / Loss Breakdown ────────────────────────────────");
            Console.WriteLine($"  Win. trades           {m.WinTrades,8}    Los. trades            {m.LossTrades,8}");
            Console.WriteLine($"  Avg. win         {m.AvgWin,14:N0}    Avg. loss          {m.AvgLoss,11:N0}");
            Console.WriteLine($"  Largest win      {m.LargestWin,14:N0}    Largest loss       {m.LargestLoss,11:N0}");
            Console.WriteLine($"  Max conseq.win        {m.MaxConseqWin,8}    Max conseq.loss        {m.MaxConseqLoss,8}");
            Console.WriteLine();
            Console.WriteLine("  ─── Returns ─────────────────────────────────────────────");
            Console.WriteLine($"  Total return       {m.TotalReturnPct,8:F2} %");
            Console.WriteLine($"  Annual return      {m.AnnualReturnPct,8:F2} %");
            Console.WriteLine($"  Max DD (abs)     {m.MaxDdAbs,14:N0}");
            Console.WriteLine(sep);
            Console.WriteLine();
        }

        public static void PrintYearlyGrid(List<Trade> trades, float capital)
        {
            if (trades == null || trades.Count == 0) return;

            var grouped = trades.GroupBy(t => t.ExitTime.Year).OrderBy(g => g.Key);
            
            Console.WriteLine(new string('=', 80));
            Console.WriteLine("  YEAR-BY-YEAR BREAKDOWN");
            Console.WriteLine(new string('=', 80));
            Console.WriteLine($"  {"Year",4}  {"#Tr",4}  {"Net PnL",12}  {"PF",5}  {"WR%",6}  {"Avg trade",10}  {"MaxDD%",10}");
            Console.WriteLine("  " + new string('-', 76));

            float runningEquity = capital;
            float runningPeak = capital;

            foreach (var g in grouped)
            {
                int year = g.Key;
                var yearTrades = g.ToList();

                int tr = yearTrades.Count;
                float gp = yearTrades.Where(t => t.PnlNet > 0).Sum(t => t.PnlNet);
                float gl = Math.Abs(yearTrades.Where(t => t.PnlNet < 0).Sum(t => t.PnlNet));
                float net = gp - gl;
                
                int wins = yearTrades.Count(t => t.PnlNet > 0);
                float wr = tr > 0 ? (float)wins / tr * 100f : 0f;
                float pf = gl > 0 ? gp / gl : float.PositiveInfinity;
                float avg = tr > 0 ? net / tr : 0f;

                // Max DD within the year (continuing equity curve from start of test)
                float yearMaxDdPct = 0f;
                foreach (var t in yearTrades)
                {
                    runningEquity += t.PnlNet;
                    if (runningEquity > runningPeak) runningPeak = runningEquity;
                    float dd = (runningPeak - runningEquity) / runningPeak * 100f;
                    if (dd > yearMaxDdPct) yearMaxDdPct = dd;
                }

                string pfStr = pf > 100 ? "  inf" : $"{pf,5:F2}";

                Console.WriteLine($"  {year,4}  {tr,4}  {net,12:N0}  {pfStr}  {wr,6:F1}%  {avg,10:N0}  {yearMaxDdPct,9:F1}%");
            }
            Console.WriteLine(new string('=', 80));
            Console.WriteLine();
        }

        public static void PrintMonthlyGrid(List<Trade> trades, float capital)
        {
            if (trades == null || trades.Count == 0) return;

            Console.WriteLine(new string('=', 115));
            Console.WriteLine("  MONTHLY RETURNS (%)");
            Console.WriteLine(new string('=', 115));
            
            string[] monthsOrder = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
            
            string header = $"{"Year",5} |";
            foreach (var m in monthsOrder) header += $"{m,7} |";
            header += $"{"Year%",9} |{"MaxDD%",9} |";
            
            Console.WriteLine(header);
            Console.WriteLine(new string('-', header.Length));

            // Generate equity curve to get monthly start/end and drawdowns
            int minYear = trades.Min(t => t.ExitTime.Year);
            int maxYear = trades.Max(t => t.ExitTime.Year);

            // Group trades by Year-Month
            var dict = trades.GroupBy(t => new { t.ExitTime.Year, t.ExitTime.Month })
                             .ToDictionary(g => g.Key, g => g.Sum(t => t.PnlNet));

            float eomEquity = capital;
            float maxPeak = capital;

            // We need to trace day by day or trade by trade to get exact BOM and EOM equity, and Max DD per month.
            // Simplified approach: iterate through all months. Find trades in that month.
            // Calculate BOM equity. Apply trades. Track Peak and DD. Calculate EOM equity.
            
            int tradeIdx = 0;
            
            for (int y = minYear; y <= maxYear; y++)
            {
                float yearBOMEquity = eomEquity; // Equity at start of year
                float yearMaxDd = 0f;
                
                string line = $"{y,5} |";

                for (int m = 1; m <= 12; m++)
                {
                    float bomEquity = eomEquity;
                    float monthMaxDd = 0f;
                    float monthlyPnl = 0f;

                    // Process trades for this month
                    while (tradeIdx < trades.Count && trades[tradeIdx].ExitTime.Year == y && trades[tradeIdx].ExitTime.Month == m)
                    {
                        var t = trades[tradeIdx];
                        monthlyPnl += t.PnlNet;
                        eomEquity += t.PnlNet;
                        
                        if (eomEquity > maxPeak) maxPeak = eomEquity;
                        float currentDd = (maxPeak - eomEquity) / maxPeak * 100f;
                        
                        if (currentDd > monthMaxDd) monthMaxDd = currentDd;
                        if (currentDd > yearMaxDd) yearMaxDd = currentDd;
                        
                        tradeIdx++;
                    }

                    float monthRet = bomEquity > 0 ? (monthlyPnl / bomEquity) * 100f : 0f;

                    if (monthlyPnl == 0f)
                        line += $"{"0.00",7} |";
                    else
                        line += $"{monthRet,7:F2} |";
                }

                float yearRet = yearBOMEquity > 0 ? (eomEquity / yearBOMEquity - 1) * 100f : 0f;
                line += $"{yearRet,8:F2}% |{yearMaxDd,8:F2}% |";
                
                Console.WriteLine(line);
            }
            Console.WriteLine(new string('-', header.Length));
            Console.WriteLine();
        }
    }
}
