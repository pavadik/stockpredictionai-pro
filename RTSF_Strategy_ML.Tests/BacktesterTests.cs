using System;
using System.Collections.Generic;
using RTSF_Strategy_ML.Core.Enums;
using RTSF_Strategy_ML.Core.Models;
using RTSF_Strategy_ML.Strategy;
using Xunit;

namespace RTSF_Strategy_ML.Tests
{
    public class BacktesterTests
    {
        [Fact]
        public void SimulateCombinedTrades_SimpleLongTrade()
        {
            var pLong = new StrategyParams { ExitDay = 1, SdelDay = 0, Leverage = 1f, MaxContracts = 0 };
            var pShort = new StrategyParams { ExitDay = 1, SdelDay = 0, Leverage = 1f, MaxContracts = 0 };

            var rowsLong = new List<StrategyDataRow>
            {
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 10, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = true, Close = 100, Contracts = 1 },
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 11, 0, 0), AllowTrade = true, InTimeWindow = true, ExitSignal = true, Close = 110, Contracts = 1 }
            };

            var rowsShort = new List<StrategyDataRow>
            {
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 10, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = false, Close = 100, Contracts = 1 },
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 11, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = false, Close = 110, Contracts = 1 }
            };

            var trades = Backtester.SimulateCombinedTrades(rowsLong, rowsShort, pLong, pShort, Backtester.FlipMode.CloseLoss, commissionPct: 0f);

            Assert.Single(trades);
            Assert.Equal(TradeDirection.Long, trades[0].Direction);
            Assert.Equal(100f, trades[0].EntryPrice);
            Assert.Equal(110f, trades[0].ExitPrice);
            Assert.Equal(10f, trades[0].Pnl1c);
            Assert.Equal("signal_reverse", trades[0].ExitReason);
            Assert.Equal(1, trades[0].Contracts);
        }

        [Fact]
        public void SimulateCombinedTrades_FlipMode_CloseLoss()
        {
            var pLong = new StrategyParams { ExitDay = 1, SdelDay = 0, Leverage = 1f, MaxContracts = 0 };
            var pShort = new StrategyParams { ExitDay = 1, SdelDay = 0, Leverage = 1f, MaxContracts = 0 };

            // Scenario: Long position opened, goes into loss, then Short entry signal fires.
            var rowsLong = new List<StrategyDataRow>
            {
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 10, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = true, Close = 100, Contracts = 1 },
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 11, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = false, Close = 90, Contracts = 1 }
            };

            var rowsShort = new List<StrategyDataRow>
            {
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 10, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = false, Close = 100, Contracts = 1 },
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 11, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = true, Close = 90, Contracts = 1 } // Opposite signal when LONG is losing (-10)
            };

            var trades = Backtester.SimulateCombinedTrades(rowsLong, rowsShort, pLong, pShort, Backtester.FlipMode.CloseLoss, commissionPct: 0f);

            // With CloseLoss, the Long position should close because it's at a loss. But it does NOT reopen a short position (because it's not a full Flip, just CloseLoss).
            Assert.Single(trades);
            Assert.Equal(TradeDirection.Long, trades[0].Direction);
            Assert.Equal(-10f, trades[0].Pnl1c);
            Assert.Equal("close_opposing", trades[0].ExitReason);
        }

        [Fact]
        public void SimulateCombinedTrades_EndOfDayExit()
        {
            var pLong = new StrategyParams { ExitDay = 1, SdelDay = 0, Leverage = 1f, MaxContracts = 0 };
            var pShort = new StrategyParams { ExitDay = 1, SdelDay = 0, Leverage = 1f, MaxContracts = 0 };

            var rowsLong = new List<StrategyDataRow>
            {
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 10, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = true, Close = 100, Contracts = 1 },
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 23, 0, 0), AllowTrade = true, InTimeWindow = false, EntrySignal = false, Close = 120, Contracts = 1 } // Out of window
            };

            var rowsShort = new List<StrategyDataRow>
            {
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 10, 0, 0), AllowTrade = true, InTimeWindow = true, EntrySignal = false, Close = 100, Contracts = 1 },
                new StrategyDataRow { Time = new DateTime(2016, 1, 1, 23, 0, 0), AllowTrade = true, InTimeWindow = false, EntrySignal = false, Close = 120, Contracts = 1 }
            };

            var trades = Backtester.SimulateCombinedTrades(rowsLong, rowsShort, pLong, pShort, Backtester.FlipMode.CloseLoss, commissionPct: 0f);

            Assert.Single(trades);
            Assert.Equal("end_of_day", trades[0].ExitReason);
            Assert.Equal(20f, trades[0].Pnl1c);
        }
    }
}