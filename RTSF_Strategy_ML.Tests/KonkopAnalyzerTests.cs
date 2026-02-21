using System;
using System.Collections.Generic;
using RTSF_Strategy_ML.Core.Enums;
using RTSF_Strategy_ML.Core.Models;
using RTSF_Strategy_ML.Reporting;
using Xunit;

namespace RTSF_Strategy_ML.Tests
{
    public class KonkopAnalyzerTests
    {
        [Fact]
        public void Analyze_ReturnsCorrectMetrics()
        {
            var trades = new List<Trade>
            {
                new Trade { EntryTime = new DateTime(2016, 1, 1), ExitTime = new DateTime(2016, 1, 2), PnlNet = 1000 },
                new Trade { EntryTime = new DateTime(2016, 1, 3), ExitTime = new DateTime(2016, 1, 4), PnlNet = -500 },
                new Trade { EntryTime = new DateTime(2016, 1, 5), ExitTime = new DateTime(2016, 1, 6), PnlNet = 2000 }
            };

            float capital = 10000;
            var m = KonkopAnalyzer.Analyze(trades, "Test Label", capital);

            Assert.Equal("Test Label", m.Label);
            Assert.Equal(3, m.TotalTrades);
            Assert.Equal(2, m.WinTrades);
            Assert.Equal(1, m.LossTrades);

            Assert.Equal(2500, m.NetProfit);
            Assert.Equal(3000, m.GrossProfit);
            Assert.Equal(500, m.GrossLoss);

            Assert.Equal(6, m.ProfitFactor); // 3000 / 500
            Assert.Equal(66.66f, m.WinRate, 1); // 2/3 * 100

            Assert.Equal(12500, m.EndEquity); // 10000 + 2500
            Assert.Equal(25f, m.TotalReturnPct); // 2500 / 10000 * 100
        }

        [Fact]
        public void Analyze_HandlesEmptyTrades()
        {
            var m = KonkopAnalyzer.Analyze(new List<Trade>(), "Empty", 10000);
            Assert.Equal(0, m.TotalTrades);
            Assert.Equal(0, m.NetProfit);
        }
    }
}
