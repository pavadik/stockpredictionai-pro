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
            float commissionPct = 0f,
            int pointValue = 1,
            float tpAtrMult = 0f,
            float tpPct = 0f)
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
            float entryAtr = 0f;
            bool tpHit = false;
            float mfe1c = 0f;
            float mae1c = 0f;
            int barsInTrade = 0;

            DateTime? currentDate = null;
            bool oneEntryTodayLong = false;
            bool oneEntryTodayShort = false;

            int tradeIdCounter = 1;

            Action<DateTime, float, string, int?> ClosePosition = (exitTs, exitPrice, reason, closeSize) =>
            {
                int sz = closeSize ?? posSize;
                float sign = posDir == TradeDirection.Short ? -1f : 1f;
                float pnl1c = sign * (exitPrice - entryPrice);
                float pnlNet = pnl1c * sz;

                trades.Add(new Trade
                {
                    Id = tradeIdCounter++,
                    Direction = posDir,
                    EntryTime = entryTime,
                    ExitTime = exitTs,
                    EntryPrice = entryPrice,
                    ExitPrice = exitPrice,
                    Contracts = sz,
                    Pnl1c = pnl1c,
                    PnlNet = pnlNet,
                    Mfe1c = mfe1c,
                    Mae1c = mae1c,
                    BarsHeld = barsInTrade,
                    ExitReason = reason
                });

                if (closeSize.HasValue)
                {
                    posSize -= sz;
                    if (posSize <= 0)
                    {
                        posDir = TradeDirection.Flat;
                        posSize = 0;
                    }
                }
                else
                {
                    posDir = TradeDirection.Flat;
                    posSize = 0;
                    barsInTrade = 0;
                }
            };

            Action<TradeDirection, DateTime, float, int, StrategyParams, float> OpenPosition = (dir, ts, price, baseLots, p, atr) =>
            {
                posDir = dir;
                int leveraged = Math.Max((int)(baseLots * p.Leverage), 1);
                if (p.MaxContracts > 0)
                    leveraged = Math.Min(leveraged, p.MaxContracts);
                posSize = leveraged;
                entryPrice = price;
                entryTime = ts;
                entryAtr = atr;
                tpHit = false;
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
                    ClosePosition(ev.Time, row.Close, "expiration", null);
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
                            
                            OpenPosition(src, ev.Time, row.Close, row.Contracts, p, row.Atr);
                            
                            if (src == TradeDirection.Long) oneEntryTodayLong = true;
                            else oneEntryTodayShort = true;
                        }
                    }
                    else if (src == posDir)
                    {
                        // Same direction bar -> check signal reverse (exit signal)
                        if (row.ExitSignal)
                        {
                            ClosePosition(ev.Time, row.Close, "signal_reverse", null);
                        }
                        else if (tpAtrMult > 0 && tpPct > 0 && !tpHit && entryAtr > 0)
                        {
                            float pnlSign = posDir == TradeDirection.Long ? 1f : -1f;
                            float unrealized = pnlSign * (row.Close - entryPrice);
                            if (unrealized >= entryAtr * tpAtrMult)
                            {
                                int tpSize = Math.Max((int)(posSize * tpPct), 1);
                                if (tpSize >= posSize)
                                    tpSize = posSize - 1;
                                if (tpSize > 0)
                                {
                                    ClosePosition(ev.Time, row.Close, "partial_tp", tpSize);
                                    tpHit = true;
                                }
                            }
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
                                ClosePosition(ev.Time, row.Close, reason, null);
                                
                                if (doReopen)
                                {
                                    OpenPosition(src, ev.Time, row.Close, row.Contracts, p, row.Atr);
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
                    ClosePosition(ev.Time, row.Close, "end_of_day", null);
                }

                // 4. End-of-week exit
                if (posSize > 0 && src == posDir && p.ExitWeek == 1 && row.IsWeekEnd)
                {
                    ClosePosition(ev.Time, row.Close, "end_of_week", null);
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
                    ClosePosition(lastRow.Time, lastRow.Close, "end_of_data", null);
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
