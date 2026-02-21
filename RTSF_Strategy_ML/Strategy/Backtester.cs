using System;
using System.Collections.Generic;
using System.Linq;
using RTSF_Strategy_ML.Core.Enums;
using RTSF_Strategy_ML.Core.Models;

namespace RTSF_Strategy_ML.Strategy
{
    public class Backtester
    {
        public enum FlipMode
        {
            Ignore,
            Flip,
            CloseOnly,
            FlipProfit,
            CloseLoss,
            FlipLongCloseShort
        }

        /// <summary>
        /// Simulates combined LONG and SHORT trades using the 'close_loss' (or other) flip logic.
        /// </summary>
        public static List<Trade> SimulateCombinedTrades(
            List<StrategyDataRow> rowsLong,
            List<StrategyDataRow> rowsShort,
            StrategyParams paramsLong,
            StrategyParams paramsShort,
            FlipMode flipMode = FlipMode.CloseLoss,
            float commissionPerContract = 0f, // e.g., slippage + commission
            int pointValue = 1) // e.g., for RTS usually we trade in points, but RTSF point value might be different. Let's just track raw points first.
        {
            var events = new List<CombinedEvent>(rowsLong.Count + rowsShort.Count);

            for (int i = 0; i < rowsLong.Count; i++)
            {
                events.Add(new CombinedEvent { Time = rowsLong[i].Time, Direction = TradeDirection.Long, Index = i, Row = rowsLong[i], Params = paramsLong });
            }
            for (int i = 0; i < rowsShort.Count; i++)
            {
                events.Add(new CombinedEvent { Time = rowsShort[i].Time, Direction = TradeDirection.Short, Index = i, Row = rowsShort[i], Params = paramsShort });
            }

            // Sort by time, maintaining order
            events.Sort((a, b) => a.Time.CompareTo(b.Time));

            var trades = new List<Trade>();
            
            TradeDirection posDir = TradeDirection.Flat;
            int posSize = 0;
            float entryPrice = 0f;
            DateTime entryTime = DateTime.MinValue;
            float mfe1c = 0f;
            float mae1c = 0f;
            int barsInTrade = 0;

            DateTime? currentDate = null;
            bool oneEntryTodayLong = false;
            bool oneEntryTodayShort = false;

            int tradeIdCounter = 1;

            Action<DateTime, float, string> ClosePosition = (exitTs, exitPrice, reason) =>
            {
                float sign = posDir == TradeDirection.Short ? -1f : 1f;
                float pnl1c = sign * (exitPrice - entryPrice);
                float pnlNet = (pnl1c - commissionPerContract) * posSize;

                trades.Add(new Trade
                {
                    Id = tradeIdCounter++,
                    Direction = posDir,
                    EntryTime = entryTime,
                    ExitTime = exitTs,
                    EntryPrice = entryPrice,
                    ExitPrice = exitPrice,
                    Contracts = posSize,
                    Pnl1c = pnl1c,
                    PnlNet = pnlNet,
                    Mfe1c = mfe1c,
                    Mae1c = mae1c,
                    BarsHeld = barsInTrade,
                    ExitReason = reason
                });

                posDir = TradeDirection.Flat;
                posSize = 0;
                barsInTrade = 0;
            };

            Action<TradeDirection, DateTime, float, int> OpenPosition = (dir, ts, price, lots) =>
            {
                posDir = dir;
                posSize = lots;
                entryPrice = price;
                entryTime = ts;
                mfe1c = 0f;
                mae1c = 0f;
                barsInTrade = 0;
            };

            foreach (var ev in events)
            {
                var row = ev.Row;
                var p = ev.Params;
                var src = ev.Direction;

                DateTime barDate = ev.Time.Date;
                if (currentDate != barDate)
                {
                    currentDate = barDate;
                    oneEntryTodayLong = false;
                    oneEntryTodayShort = false;
                }

                if (posSize > 0)
                {
                    barsInTrade++;
                    // Update MFE / MAE
                    float sign = posDir == TradeDirection.Short ? -1f : 1f;
                    float mfe = sign * (row.Close - entryPrice);
                    float mae = sign * (row.Close - entryPrice);
                    
                    if (posDir == TradeDirection.Long)
                    {
                        mfe = row.High - entryPrice;
                        mae = row.Low - entryPrice;
                    }
                    else if (posDir == TradeDirection.Short)
                    {
                        mfe = entryPrice - row.Low;
                        mae = entryPrice - row.High;
                    }

                    if (mfe > mfe1c) mfe1c = mfe;
                    if (mae < mae1c) mae1c = mae;
                }

                // 1. Forced exit: Expiration week
                if (posSize > 0 && src == posDir && !row.AllowTrade)
                {
                    ClosePosition(ev.Time, row.Close, "expiration");
                    continue;
                }

                // 2. Inside trading window
                if (row.AllowTrade && row.InTimeWindow)
                {
                    if (posSize == 0)
                    {
                        // Flat -> open if entry signal
                        if (row.EntrySignal)
                        {
                            if (p.SdelDay == 1)
                            {
                                if (src == TradeDirection.Long && oneEntryTodayLong) continue;
                                if (src == TradeDirection.Short && oneEntryTodayShort) continue;
                            }
                            
                            // Only open if contracts > 0
                            if (row.Contracts > 0)
                            {
                                OpenPosition(src, ev.Time, row.Close, row.Contracts);
                                
                                if (src == TradeDirection.Long) oneEntryTodayLong = true;
                                else oneEntryTodayShort = true;
                            }
                        }
                    }
                    else if (src == posDir)
                    {
                        // Same direction bar -> check signal reverse (exit signal)
                        if (row.ExitSignal)
                        {
                            ClosePosition(ev.Time, row.Close, "signal_reverse");
                        }
                    }
                    else
                    {
                        // Opposing direction bar -> handle based on flip_mode
                        if (row.EntrySignal && flipMode != FlipMode.Ignore)
                        {
                            if (p.SdelDay == 1)
                            {
                                if (src == TradeDirection.Long && oneEntryTodayLong) continue;
                                if (src == TradeDirection.Short && oneEntryTodayShort) continue;
                            }

                            float sign = posDir == TradeDirection.Short ? -1f : 1f;
                            float unrealised = sign * (row.Close - entryPrice);

                            bool doClose = false;
                            bool doReopen = false;

                            switch (flipMode)
                            {
                                case FlipMode.Flip:
                                    doClose = true;
                                    doReopen = true;
                                    break;
                                case FlipMode.CloseOnly:
                                    doClose = true;
                                    break;
                                case FlipMode.FlipProfit:
                                    if (unrealised > 0)
                                    {
                                        doClose = true;
                                        doReopen = true;
                                    }
                                    break;
                                case FlipMode.CloseLoss:
                                    if (unrealised <= 0)
                                    {
                                        doClose = true;
                                    }
                                    break;
                                case FlipMode.FlipLongCloseShort:
                                    if (posDir == TradeDirection.Long)
                                    {
                                        doClose = true;
                                        doReopen = true;
                                    }
                                    else
                                    {
                                        doClose = true;
                                    }
                                    break;
                            }

                            if (doClose)
                            {
                                string reason = doReopen ? "flip" : "close_opposing";
                                ClosePosition(ev.Time, row.Close, reason);
                                
                                if (doReopen)
                                {
                                    OpenPosition(src, ev.Time, row.Close, row.Contracts);
                                }
                                
                                if (src == TradeDirection.Long) oneEntryTodayLong = true;
                                else oneEntryTodayShort = true;
                            }
                        }
                    }
                }
                // 3. Outside window: end-of-day exit for current direction
                else if (posSize > 0 && src == posDir && p.ExitDay == 1)
                {
                    ClosePosition(ev.Time, row.Close, "end_of_day");
                }

                // 4. End-of-week exit
                if (posSize > 0 && src == posDir && p.ExitWeek == 1 && row.IsWeekEnd)
                {
                    ClosePosition(ev.Time, row.Close, "end_of_week");
                }
            }

            // Close any open position at end of data
            if (posSize > 0)
            {
                // Find the last row of the current active direction
                var lastRow = posDir == TradeDirection.Long 
                    ? rowsLong.LastOrDefault() 
                    : rowsShort.LastOrDefault();
                
                if (lastRow != null)
                {
                    ClosePosition(lastRow.Time, lastRow.Close, "end_of_data");
                }
            }

            return trades;
        }

        private class CombinedEvent
        {
            public DateTime Time { get; set; }
            public TradeDirection Direction { get; set; }
            public int Index { get; set; }
            public StrategyDataRow Row { get; set; } = null!;
            public StrategyParams Params { get; set; } = null!;
        }
    }
}
