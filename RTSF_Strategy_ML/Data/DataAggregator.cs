using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using RTSF_Strategy_ML.Core.Models;

namespace RTSF_Strategy_ML.Data
{
    public class DataAggregator
    {
        /// <summary>
        /// Session-aware aggregation of M1 bars to custom intraday timeframes.
        /// Groups bars within each trading day so that resulting bars never span overnight or lunch breaks.
        /// </summary>
        public static List<Bar> AggregateIntradayCustom(List<Bar> m1Bars, int nMinutes)
        {
            if (nMinutes <= 0)
                throw new ArgumentException("nMinutes must be positive", nameof(nMinutes));
            if (nMinutes == 1)
                return m1Bars.ToList();

            var result = new List<Bar>();

            // Group M1 bars by Date
            var groupedByDate = m1Bars.GroupBy(b => b.Time.Date);

            foreach (var dayGroup in groupedByDate)
            {
                var dayBars = dayGroup.ToList();
                int n = dayBars.Count;
                if (n == 0) continue;

                // Group IDs corresponding to numpy: np.arange(n) // n_minutes
                for (int i = 0; i < n; i += nMinutes)
                {
                    int endIdx = Math.Min(i + nMinutes, n) - 1;

                    float open = dayBars[i].Open;
                    float high = dayBars[i].High;
                    float low = dayBars[i].Low;
                    float close = dayBars[endIdx].Close;
                    long volume = 0;

                    for (int j = i; j <= endIdx; j++)
                    {
                        if (dayBars[j].High > high) high = dayBars[j].High;
                        if (dayBars[j].Low < low) low = dayBars[j].Low;
                        volume += dayBars[j].Volume;
                    }

                    // Resulting bar's time is the timestamp of the last M1 bar in the group
                    result.Add(new Bar(dayBars[endIdx].Time, open, high, low, close, volume));
                }
            }

            return result;
        }

        /// <summary>
        /// From raw M1 bars build the aligned multi-TF StrategyDataRow objects ready for signal generation.
        /// Replicates `prepare_strategy_data` and `pre_aggregate` python functions.
        /// </summary>
        public static List<StrategyDataRow> PrepareStrategyData(List<Bar> m1Bars, int tf1Minutes, int tf2Minutes)
        {
            var tf2Bars = AggregateIntradayCustom(m1Bars, tf2Minutes);
            var tf1Bars = AggregateIntradayCustom(m1Bars, tf1Minutes);

            // Compute first TF2 bar's time-of-day per day to calculate ElapsedMinutes
            // Python: elapsed = (hour*60+minute) - first_tf2_bar_minute_of_day (per day)
            var firstTf2MinutePerDate = tf2Bars.GroupBy(b => b.Time.Date)
                .ToDictionary(g => g.Key, g => g.First().Time.Hour * 60 + g.First().Time.Minute);

            var result = new List<StrategyDataRow>(tf2Bars.Count);
            int tf1Index = 0;

            for (int i = 0; i < tf2Bars.Count; i++)
            {
                var tf2 = tf2Bars[i];

                // As-of alignment: find the last TF1 bar whose time is <= TF2 bar's time
                while (tf1Index + 1 < tf1Bars.Count && tf1Bars[tf1Index + 1].Time <= tf2.Time)
                {
                    tf1Index++;
                }

                Bar? currentTf1 = null;
                if (tf1Bars.Count > 0 && tf1Bars[tf1Index].Time <= tf2.Time)
                {
                    currentTf1 = tf1Bars[tf1Index];
                }

                var row = new StrategyDataRow
                {
                    Time = tf2.Time,
                    Open = tf2.Open,
                    High = tf2.High,
                    Low = tf2.Low,
                    Close = tf2.Close,
                    Volume = tf2.Volume,

                    CloseTf1 = currentTf1?.Close ?? float.NaN,
                    HighTf1 = currentTf1?.High ?? float.NaN,
                    LowTf1 = currentTf1?.Low ?? float.NaN
                };

                // Expiration filter: block days 8-17 in Mar, Jun, Sep, Dec
                int day = tf2.Time.Day;
                int month = tf2.Time.Month;
                bool inExpiry = (day > 7) && (day < 18) && 
                                (month == 3 || month == 6 || month == 9 || month == 12);
                row.AllowTrade = !inExpiry;

                // Elapsed minutes from first TF2 bar of the day (matches Python _minutes_from_day_start)
                if (firstTf2MinutePerDate.TryGetValue(tf2.Time.Date, out int firstMinute))
                {
                    row.ElapsedMinutes = (tf2.Time.Hour * 60 + tf2.Time.Minute) - firstMinute;
                }

                // Pre-calculated True Range (on TF2) for ATR calculation later
                if (i == 0)
                {
                    row.TrueRange = tf2.High - tf2.Low;
                }
                else
                {
                    float prevClose = tf2Bars[i - 1].Close;
                    float tr1 = tf2.High - tf2.Low;
                    float tr2 = Math.Abs(tf2.High - prevClose);
                    float tr3 = Math.Abs(tf2.Low - prevClose);
                    row.TrueRange = Math.Max(tr1, Math.Max(tr2, tr3));
                }

                result.Add(row);
            }

            // Mark week-ends (is_week_end flag equivalent to np.append(wid[:-1] != wid[1:], [True]))
            for (int i = 0; i < result.Count - 1; i++)
            {
                var curDate = result[i].Time;
                var nextDate = result[i + 1].Time;

                int curYear = ISOWeek.GetYear(curDate);
                int curWeek = ISOWeek.GetWeekOfYear(curDate);
                
                int nextYear = ISOWeek.GetYear(nextDate);
                int nextWeek = ISOWeek.GetWeekOfYear(nextDate);

                int curWid = curYear * 100 + curWeek;
                int nextWid = nextYear * 100 + nextWeek;

                result[i].IsWeekEnd = (curWid != nextWid);
            }
            if (result.Count > 0)
            {
                result[^1].IsWeekEnd = true;
            }

            return result;
        }
    }
}
