using System;
using System.Collections.Generic;
using System.Linq;
using RTSF_Strategy_ML.Core;
using RTSF_Strategy_ML.Core.Enums;
using RTSF_Strategy_ML.Core.Models;

namespace RTSF_Strategy_ML.ML
{
    public class FeatureBuilder
    {
        private readonly List<Bar> _m1Bars;
        private readonly Dictionary<DateTime, int> _m1TimeToIndex;

        private readonly List<Bar> _d1Bars;
        private readonly Dictionary<DateTime, int> _d1DateToIndex;

        private readonly float[] _d1Tr;
        private readonly float[] _d1Adx;

        public FeatureBuilder(List<Bar> m1Bars)
        {
            _m1Bars = m1Bars.OrderBy(b => b.Time).ToList();
            _m1TimeToIndex = new Dictionary<DateTime, int>(_m1Bars.Count);
            
            for (int i = 0; i < _m1Bars.Count; i++)
            {
                _m1TimeToIndex[_m1Bars[i].Time] = i;
            }

            // Build D1 bars
            _d1Bars = AggregateToD1(_m1Bars);
            _d1DateToIndex = new Dictionary<DateTime, int>(_d1Bars.Count);
            for (int i = 0; i < _d1Bars.Count; i++)
            {
                _d1DateToIndex[_d1Bars[i].Time.Date] = i;
            }

            // Precalculate D1 features
            _d1Tr = Indicators.TrueRange(_d1Bars);
            _d1Adx = Indicators.Adx(_d1Bars, 14);
        }

        private List<Bar> AggregateToD1(List<Bar> m1Bars)
        {
            var result = new List<Bar>();
            var grouped = m1Bars.GroupBy(b => b.Time.Date);

            foreach (var g in grouped)
            {
                var bars = g.ToList();
                float open = bars.First().Open;
                float close = bars.Last().Close;
                float high = bars.Max(b => b.High);
                float low = bars.Min(b => b.Low);
                long volume = bars.Sum(b => b.Volume);

                result.Add(new Bar(g.Key, open, high, low, close, volume));
            }

            return result;
        }

        public float[] GetFeatures(DateTime entryTime, TradeDirection direction)
        {
            float[] features = new float[14];

            // 1. hour
            features[0] = entryTime.Hour;
            // 2. minute
            features[1] = entryTime.Minute;
            // 3. day_of_week (Python weekday(): Monday=0, Sunday=6)
            int dayOfWeek = (int)entryTime.DayOfWeek - 1;
            if (dayOfWeek < 0) dayOfWeek = 6; // Sunday is 0 in C#, but 6 in Python
            features[2] = dayOfWeek;
            // 4. month
            features[3] = entryTime.Month;

            // --- Macro / D1 Features ---
            DateTime entryDate = entryTime.Date;
            if (_d1DateToIndex.TryGetValue(entryDate, out int d1Loc) && d1Loc > 10)
            {
                int yestLoc = d1Loc - 1;
                var yestBar = _d1Bars[yestLoc];
                
                // 5. d1_volume
                features[4] = yestBar.Volume;
                // 6. d1_tr
                features[5] = _d1Tr[yestLoc];
                // 7. d1_adx
                features[6] = _d1Adx[yestLoc];
                // 8. d1_ret_1d
                features[7] = (yestBar.Close / _d1Bars[yestLoc - 1].Close) - 1.0f;
                // 9. d1_ret_5d
                features[8] = (yestBar.Close / _d1Bars[yestLoc - 5].Close) - 1.0f;
            }
            else
            {
                features[4] = 0f;
                features[5] = 0f;
                features[6] = 0f;
                features[7] = 0f;
                features[8] = 0f;
            }

            // --- Micro / M1 Features ---
            if (_m1TimeToIndex.TryGetValue(entryTime, out int m1Loc) && m1Loc > 120)
            {
                var entryBar = _m1Bars[m1Loc];
                // 10. m1_volume
                features[9] = entryBar.Volume;
                // 11. m1_ret_15m
                features[10] = (entryBar.Close / _m1Bars[m1Loc - 15].Close) - 1.0f;
                // 12. m1_ret_60m
                features[11] = (entryBar.Close / _m1Bars[m1Loc - 60].Close) - 1.0f;
                // 13. m1_ret_120m
                features[12] = (entryBar.Close / _m1Bars[m1Loc - 120].Close) - 1.0f;
            }
            else
            {
                features[9] = 0f;
                features[10] = 0f;
                features[11] = 0f;
                features[12] = 0f;
            }

            // 14. is_long
            features[13] = direction == TradeDirection.Long ? 1f : 0f;

            return features;
        }
    }
}
