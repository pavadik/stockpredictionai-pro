using System;
using System.Collections.Generic;
using RTSF_Strategy_ML.Core.Models;
using RTSF_Strategy_ML.Strategy;
using RTSF_Strategy_ML.Core.Enums;
using Xunit;

namespace RTSF_Strategy_ML.Tests
{
    public class MomentumTrendStrategyTests
    {
        [Fact]
        public void GenerateSignals_CalculatesCorrectly()
        {
            var p = new StrategyParams
            {
                Lookback = 10,
                Length = 20,
                Lookback2 = 5,
                Length2 = 10,
                MinS = 0,
                MaxS = 1440,
                Koeff1 = 1.0f,
                Koeff2 = 1.0f,
                Direction = TradeDirection.Long,
                Mmcoff = 5,
                Capital = 5000000,
                Leverage = 1.0f,
                MaxContracts = 100
            };

            var strategy = new MomentumTrendStrategy(p);

            var rows = new List<StrategyDataRow>();

            // Generate enough rows to trigger signals
            for (int i = 0; i < 50; i++)
            {
                rows.Add(new StrategyDataRow
                {
                    Time = new DateTime(2016, 1, 1, 10, 0, 0).AddMinutes(i * 90),
                    CloseTf1 = 100 + i * 2, // Up trend
                    Close = 100 + i * 2,
                    TrueRange = 2,
                    ElapsedMinutes = i * 90 % 1440,
                    AllowTrade = true,
                    InTimeWindow = true
                });
            }

            strategy.GenerateSignals(rows);

            // Row 0-19 will have NaN for TF1 SMA and TSA (length=20)
            for (int i = 0; i < 19; i++)
            {
                Assert.False(rows[i].EntrySignal);
            }

            // By row 30, TF1 and TF2 should be fully valid and in up trend
            // Since close is monotonically increasing, TrendScore > 0, SMA is lagging (so Close > SMA), TSA is lagging (so TrendScore > TSA).
            // Thus condTf1 and condTf2 should both be true.
            
            bool foundEntry = false;
            for (int i = 20; i < 50; i++)
            {
                if (rows[i].EntrySignal)
                {
                    foundEntry = true;
                    break;
                }
            }

            Assert.True(foundEntry, "Should find at least one entry signal in an up trend");

            // Check position sizing
            // ATR with TR=2 should be 2.
            // Capital=5000000, PointValueMult=300, ATR=2 -> 5000000 / (300*2) = 8333.33 -> 8333 -> capped at MaxContracts=100
            Assert.Equal(100, rows[49].Contracts);
        }

        [Fact]
        public void GenerateSignals_ShortDirection_Works()
        {
            var p = new StrategyParams
            {
                Lookback = 5,
                Length = 10,
                Lookback2 = 5,
                Length2 = 10,
                MinS = 0,
                MaxS = 1440,
                Koeff1 = 1.0f,
                Koeff2 = 1.0f,
                Direction = TradeDirection.Short, // SHORT
                Mmcoff = 5,
                Capital = 5000000,
                Leverage = 1.0f,
                MaxContracts = 100
            };

            var strategy = new MomentumTrendStrategy(p);

            var rows = new List<StrategyDataRow>();

            // Down trend
            for (int i = 0; i < 50; i++)
            {
                rows.Add(new StrategyDataRow
                {
                    Time = new DateTime(2016, 1, 1, 10, 0, 0).AddMinutes(i * 90),
                    CloseTf1 = 200 - i * 2, 
                    Close = 200 - i * 2,
                    TrueRange = 2,
                    ElapsedMinutes = i * 90 % 1440,
                    AllowTrade = true,
                    InTimeWindow = true
                });
            }

            strategy.GenerateSignals(rows);
            
            bool foundEntry = false;
            for (int i = 10; i < 50; i++)
            {
                if (rows[i].EntrySignal)
                {
                    foundEntry = true;
                    break;
                }
            }

            Assert.True(foundEntry, "Should find at least one SHORT entry signal in a down trend");
        }
    }
}