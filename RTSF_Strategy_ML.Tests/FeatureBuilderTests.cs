using System;
using System.Collections.Generic;
using RTSF_Strategy_ML.Core.Enums;
using RTSF_Strategy_ML.Core.Models;
using RTSF_Strategy_ML.ML;
using Xunit;

namespace RTSF_Strategy_ML.Tests
{
    public class FeatureBuilderTests
    {
        [Fact]
        public void GetFeatures_BasicTimeFeatures_AreCorrect()
        {
            // Just a minimal set of bars
            var bars = new List<Bar>
            {
                new Bar(new DateTime(2016, 1, 4, 10, 0, 0), 100, 105, 95, 100, 1000)
            };

            var fb = new FeatureBuilder(bars);
            var entryTime = new DateTime(2016, 1, 4, 10, 0, 0); // Monday, Jan 4th
            
            var features = fb.GetFeatures(entryTime, TradeDirection.Long);

            Assert.Equal(14, features.Length);
            
            // 1. hour
            Assert.Equal(10f, features[0]);
            // 2. minute
            Assert.Equal(0f, features[1]);
            // 3. day_of_week (Monday = 0 in Python)
            Assert.Equal(0f, features[2]);
            // 4. month
            Assert.Equal(1f, features[3]);
            // 14. is_long
            Assert.Equal(1f, features[13]);
        }

        [Fact]
        public void GetFeatures_ReturnsZeros_IfNotEnoughHistory()
        {
            var bars = new List<Bar>
            {
                new Bar(new DateTime(2016, 1, 4, 10, 0, 0), 100, 105, 95, 100, 1000)
            };

            var fb = new FeatureBuilder(bars);
            var entryTime = new DateTime(2016, 1, 4, 10, 0, 0);
            
            var features = fb.GetFeatures(entryTime, TradeDirection.Short);

            // Since there is no D1 history (-6 days) and no M1 history (-120 bars), they should be 0
            Assert.Equal(0f, features[4]); // d1_volume
            Assert.Equal(0f, features[5]); // d1_tr
            Assert.Equal(0f, features[6]); // d1_adx
            Assert.Equal(0f, features[7]); // d1_ret_1d
            Assert.Equal(0f, features[8]); // d1_ret_5d
            
            Assert.Equal(0f, features[9]); // m1_volume
            Assert.Equal(0f, features[10]); // m1_ret_15m
            Assert.Equal(0f, features[11]); // m1_ret_60m
            Assert.Equal(0f, features[12]); // m1_ret_120m
            
            // 14. is_long
            Assert.Equal(0f, features[13]);
        }
    }
}
