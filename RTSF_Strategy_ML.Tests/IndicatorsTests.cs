using System;
using System.Collections.Generic;
using RTSF_Strategy_ML.Core;
using RTSF_Strategy_ML.Core.Models;
using Xunit;

namespace RTSF_Strategy_ML.Tests
{
    public class IndicatorsTests
    {
        [Fact]
        public void Sma_Basic_CalculatesCorrectly()
        {
            float[] series = { 1, 2, 3, 4, 5 };
            // min_periods=1: partial averages from bar 0
            // i=0: avg(1) = 1.0
            // i=1: avg(1,2) = 1.5
            // i=2: avg(1,2,3) = 2.0
            // i=3: avg(2,3,4) = 3.0
            // i=4: avg(3,4,5) = 4.0

            var result = Indicators.Sma(series, 3);

            Assert.Equal(5, result.Length);
            Assert.Equal(1.0f, result[0]);
            Assert.Equal(1.5f, result[1]);
            Assert.Equal(2.0f, result[2]);
            Assert.Equal(3.0f, result[3]);
            Assert.Equal(4.0f, result[4]);
        }

        [Fact]
        public void Sma_WithNaNs_IgnoresNaNs()
        {
            float[] series = { 1, 2, float.NaN, 4, 5 };
            // Period 3, min_periods=1
            // i=0 (vals: 1): sum=1, count=1 -> 1.0
            // i=1 (vals: 1, 2): sum=3, count=2 -> 1.5
            // i=2 (vals: 1, 2, NaN): sum=3, count=2 -> 1.5
            // i=3 (vals: 2, NaN, 4): sum=6, count=2 -> 3.0
            // i=4 (vals: NaN, 4, 5): sum=9, count=2 -> 4.5
            var result = Indicators.Sma(series, 3);

            Assert.Equal(1.0f, result[0]);
            Assert.Equal(1.5f, result[1]);
            Assert.Equal(1.5f, result[2]);
            Assert.Equal(3.0f, result[3]);
            Assert.Equal(4.5f, result[4]);
        }

        [Fact]
        public void Ema_Basic_CalculatesCorrectly()
        {
            float[] series = { 1, 2, 3, 4, 5 };
            int span = 3; // alpha = 2 / 4 = 0.5
            
            var result = Indicators.Ema(series, span);
            
            // Expected: 
            // result[0] = 1
            // result[1] = 0.5 * 2 + 0.5 * 1 = 1.5
            // result[2] = 0.5 * 3 + 0.5 * 1.5 = 2.25
            // result[3] = 0.5 * 4 + 0.5 * 2.25 = 3.125
            
            Assert.Equal(1.0f, result[0]);
            Assert.Equal(1.5f, result[1]);
            Assert.Equal(2.25f, result[2]);
            Assert.Equal(3.125f, result[3]);
        }

        [Fact]
        public void TrueRange_CalculatesCorrectly()
        {
            var bars = new List<Bar>
            {
                new Bar(DateTime.Now, 10, 15, 8, 12, 100), // tr: 15 - 8 = 7
                new Bar(DateTime.Now, 12, 20, 10, 18, 100)  // tr: max(20-10, |20-12|, |10-12|) = 10
            };

            var tr = Indicators.TrueRange(bars);

            Assert.Equal(2, tr.Length);
            Assert.Equal(7, tr[0]);
            Assert.Equal(10, tr[1]);
        }

        [Fact]
        public void Atr_CalculatesCorrectly()
        {
            var bars = new List<Bar>
            {
                new Bar(DateTime.Now, 10, 15, 8, 12, 100), // TR=7
                new Bar(DateTime.Now, 12, 20, 10, 18, 100), // TR=10
                new Bar(DateTime.Now, 18, 25, 15, 20, 100)  // TR=10 (max(25-15, |25-18|, |15-18|))
            };

            var atr = Indicators.Atr(bars, 2);

            // min_periods=1 logic:
            // i=0: TR=7 -> atr[0]=7
            // i=1: TR=10 -> atr[1]=(7+10)/2 = 8.5
            // i=2: TR=10 -> atr[2]=(10+10)/2 = 10
            Assert.Equal(7.0f, atr[0]);
            Assert.Equal(8.5f, atr[1]);
            Assert.Equal(10.0f, atr[2]);
        }

        [Fact]
        public void TrendScore_CalculatesCorrectly()
        {
            // Close = [ 10, 11, 12, 13, ... up to 20 elements ]
            float[] close = new float[20];
            for (int i = 0; i < 20; i++) close[i] = i * 10; // strictly increasing

            int lookback = 5;
            var score = Indicators.TrendScore(close, lookback);

            // shift 1 to 10. For i=19, lookback=5, shifts are 6 to 15.
            // close[19] is always > close[19 - shift], so diff is > 0, Math.Sign(diff) = 1.
            // Sum of 10 values = 10.
            Assert.Equal(10.0f, score[19]);
        }
    }
}