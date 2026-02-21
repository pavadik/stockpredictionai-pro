using System;
using System.Collections.Generic;
using RTSF_Strategy_ML.Core.Models;
using RTSF_Strategy_ML.Data;
using Xunit;

namespace RTSF_Strategy_ML.Tests
{
    public class DataAggregatorTests
    {
        [Fact]
        public void AggregateIntradayCustom_WorksProperly()
        {
            var bars = new List<Bar>
            {
                new Bar(new DateTime(2016, 1, 1, 10, 0, 0), 100, 105, 95, 101, 1000),
                new Bar(new DateTime(2016, 1, 1, 10, 1, 0), 101, 106, 96, 102, 1000),
                new Bar(new DateTime(2016, 1, 1, 10, 2, 0), 102, 107, 97, 103, 1000),
                new Bar(new DateTime(2016, 1, 1, 10, 3, 0), 103, 108, 98, 104, 1000), // Day 1
                new Bar(new DateTime(2016, 1, 2, 10, 0, 0), 104, 109, 99, 105, 1000)  // Day 2
            };

            // Aggregate into 3-minute bars
            var agg = DataAggregator.AggregateIntradayCustom(bars, 3);

            // Day 1 has 4 bars, groups: [0, 1, 2] and [3]
            // Day 2 has 1 bar, group: [0]
            Assert.Equal(3, agg.Count);
            
            // Bar 1 (Day 1, 10:02:00)
            Assert.Equal(new DateTime(2016, 1, 1, 10, 2, 0), agg[0].Time);
            Assert.Equal(100f, agg[0].Open); // first open
            Assert.Equal(107f, agg[0].High); // max high
            Assert.Equal(95f, agg[0].Low);   // min low
            Assert.Equal(103f, agg[0].Close); // last close
            Assert.Equal(3000L, agg[0].Volume);

            // Bar 2 (Day 1, 10:03:00)
            Assert.Equal(new DateTime(2016, 1, 1, 10, 3, 0), agg[1].Time);
            Assert.Equal(103f, agg[1].Open);
            Assert.Equal(108f, agg[1].High);
            Assert.Equal(98f, agg[1].Low);
            Assert.Equal(104f, agg[1].Close);
            Assert.Equal(1000L, agg[1].Volume);
            
            // Bar 3 (Day 2, 10:00:00)
            Assert.Equal(new DateTime(2016, 1, 2, 10, 0, 0), agg[2].Time);
        }

        [Fact]
        public void PrepareStrategyData_CalculatesCorrectFlags()
        {
            var bars = new List<Bar>();
            // Create 91 M1 bars to ensure we get two 90-minute TF2 bars
            for (int i = 0; i < 91; i++)
            {
                bars.Add(new Bar(new DateTime(2016, 3, 15, 10, 0, 0).AddMinutes(i), 100, 100, 100, 100, 100));
            }

            var rows = DataAggregator.PrepareStrategyData(bars, 210, 90);

            // March 15th is an expiration day (month=3, day=15 -> 8..17 range). AllowTrade should be false.
            // 91 bars grouped by 90 -> 2 groups (0..89 and 90)
            Assert.Equal(2, rows.Count);
            Assert.False(rows[0].AllowTrade);
            Assert.False(rows[1].AllowTrade);
            
            // 1st group ends at 10:00 + 89m = 11:29
            // 2nd group ends at 10:00 + 90m = 11:30
            // ElapsedMinutes is calculated from the first M1 bar of the day (10:00) using the resulting TF2 bar time
            Assert.Equal(89, rows[0].ElapsedMinutes);
            Assert.Equal(90, rows[1].ElapsedMinutes);
        }

        [Fact]
        public void PrepareStrategyData_AsOfJoin_WorksProperly()
        {
            // TF1 = 210m, TF2 = 90m
            // 10:00, 11:30, 13:00, 14:30
            var bars = new List<Bar>
            {
                new Bar(new DateTime(2016, 1, 4, 10, 0, 0), 100, 110, 90, 105, 100), // M1 #1
                new Bar(new DateTime(2016, 1, 4, 11, 30, 0), 105, 115, 95, 110, 100), // M1 #90
                new Bar(new DateTime(2016, 1, 4, 13, 30, 0), 110, 120, 100, 115, 100), // M1 #210
                new Bar(new DateTime(2016, 1, 4, 15, 0, 0), 115, 125, 105, 120, 100), 
            };

            var rows = DataAggregator.PrepareStrategyData(bars, 210, 90);

            // Since M1 are sparse, let's just observe how the join works based on Time
            // tf2 bars will be built at [10:00, 11:30, 13:30, 15:00] (since each is <90m or >=90m from start)
            // tf1 bars built at [10:00, 11:30, 13:30, 15:00] due to sparsity.
            // Wait, AggregateIntradayCustom groups by index. Let's make continuous bars to test TF1 vs TF2 better.
            
            var continuousBars = new List<Bar>();
            for(int i=0; i<300; i++)
            {
                continuousBars.Add(new Bar(new DateTime(2016, 1, 4, 10, 0, 0).AddMinutes(i), 100+i, 100+i+5, 100+i-5, 100+i, 100));
            }

            var rowsContinuous = DataAggregator.PrepareStrategyData(continuousBars, 210, 90);

            // TF2 bars (90 mins):
            // 0..89 -> last bar at 11:29:00 (since 10:00 + 89m)
            // 90..179 -> last bar at 12:59:00
            // 180..269 -> last bar at 14:29:00
            // 270..299 -> last bar at 14:59:00

            // TF1 bars (210 mins):
            // 0..209 -> last bar at 13:29:00
            // 210..299 -> last bar at 14:59:00

            Assert.Equal(4, rowsContinuous.Count);

            // For TF2 bar 1 (11:29:00): TF1 hasn't completed any bar that's <= 11:29:00. Wait, TF1 first bar completes at 13:29:00.
            // So TF1 values should be NaN or default if none <= 11:29:00
            Assert.True(float.IsNaN(rowsContinuous[0].CloseTf1));
            Assert.True(float.IsNaN(rowsContinuous[1].CloseTf1)); // 12:59:00
            
            // For TF2 bar 3 (14:29:00): TF1 first bar completed at 13:29:00.
            Assert.Equal(100 + 209, rowsContinuous[2].CloseTf1); // The close of the 210th M1 bar
            
            // For TF2 bar 4 (14:59:00): TF1 second bar completed at 14:59:00.
            Assert.Equal(100 + 299, rowsContinuous[3].CloseTf1); 
        }
    }
}