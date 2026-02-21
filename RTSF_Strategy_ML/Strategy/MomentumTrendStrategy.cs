using System;
using System.Collections.Generic;
using System.Linq;
using RTSF_Strategy_ML.Core;
using RTSF_Strategy_ML.Core.Enums;
using RTSF_Strategy_ML.Core.Models;

namespace RTSF_Strategy_ML.Strategy
{
    public class MomentumTrendStrategy
    {
        private readonly StrategyParams _params;

        public MomentumTrendStrategy(StrategyParams parameters)
        {
            _params = parameters;
        }

        public void GenerateSignals(List<StrategyDataRow> rows)
        {
            if (rows == null || rows.Count == 0) return;
            int n = rows.Count;

            // 1. Extract raw series for indicators
            var closeTf1 = new float[n];
            var closeTf2 = new float[n];
            var trSeries = new float[n];
            
            for (int i = 0; i < n; i++)
            {
                closeTf1[i] = rows[i].CloseTf1;
                closeTf2[i] = rows[i].Close;
                trSeries[i] = rows[i].TrueRange;
            }

            // 2. Compute TF1 Indicators (Trend)
            var trendScoreTf1 = Indicators.TrendScore(closeTf1, _params.Lookback);
            var tsaTf1 = Indicators.Sma(trendScoreTf1, _params.Length);
            var smaTf1 = Indicators.Sma(closeTf1, _params.Length);

            // 3. Compute TF2 Indicators (Trading)
            var trendScoreTf2 = Indicators.TrendScore(closeTf2, _params.Lookback2);
            var tsaTf2 = Indicators.Sma(trendScoreTf2, _params.Length2);
            var smaTf2 = Indicators.Sma(closeTf2, _params.Length2);

            // 4. Compute ATR for position sizing
            var atr = new float[n];
            for (int i = 0; i < n; i++)
            {
                if (i < _params.Mmcoff - 1)
                {
                    float sum = 0;
                    for (int j = 0; j <= i; j++) sum += trSeries[j];
                    atr[i] = sum / (i + 1); // min_periods=1 behavior
                }
                else
                {
                    float sum = 0;
                    for (int j = i - _params.Mmcoff + 1; j <= i; j++) sum += trSeries[j];
                    atr[i] = sum / _params.Mmcoff;
                }
            }

            // 5. Apply conditions and assign to rows
            for (int i = 0; i < n; i++)
            {
                var r = rows[i];
                
                r.TrendScore = trendScoreTf1[i];
                r.Tsa = tsaTf1[i];
                r.SmaTf1 = smaTf1[i];

                r.TrendScore2 = trendScoreTf2[i];
                r.Tsa2 = tsaTf2[i];
                r.SmaTf2 = smaTf2[i];

                r.Atr = atr[i];

                r.InTimeWindow = r.ElapsedMinutes >= _params.MinS && r.ElapsedMinutes < _params.MaxS;
                bool active = r.AllowTrade && r.InTimeWindow;

                // Check for NaN in indicators to avoid false signals
                bool isTf1Valid = !float.IsNaN(r.Tsa) && !float.IsNaN(r.SmaTf1);
                bool isTf2Valid = !float.IsNaN(r.Tsa2) && !float.IsNaN(r.SmaTf2);

                if (_params.Direction == TradeDirection.Long)
                {
                    bool condTf1 = isTf1Valid && (r.TrendScore > r.Tsa * _params.Koeff1) && (r.CloseTf1 > r.SmaTf1 * _params.Koeff1);
                    bool condTf2 = isTf2Valid && (r.TrendScore2 > r.Tsa2 * _params.Koeff2) && (r.Close > r.SmaTf2 * _params.Koeff2);

                    r.EntrySignal = active && condTf1 && condTf2;
                    r.ExitSignal = active && condTf1 && !condTf2;
                }
                else // Short
                {
                    bool condTf1Short = isTf1Valid && (r.TrendScore < r.Tsa * (2 - _params.Koeff1)) && (r.CloseTf1 < r.SmaTf1 * (2 - _params.Koeff1));
                    bool condTf2Short = isTf2Valid && (r.TrendScore2 < r.Tsa2 * (2 - _params.Koeff2)) && (r.Close < r.SmaTf2 * (2 - _params.Koeff2));

                    r.EntrySignal = active && condTf1Short && condTf2Short;
                    r.ExitSignal = active && condTf1Short && !condTf2Short;
                }

                // Compute ATR for position sizing
                if (r.Atr > 0)
                {
                    // To match python's floor(capital / (point_value_mult * atr))
                    float rawContracts = _params.Capital / (_params.PointValueMult * r.Atr);
                    
                    // Pandas fillna(0) and astype(int).clip(lower=0)
                    r.ContractsBase = float.IsInfinity(rawContracts) || float.IsNaN(rawContracts) ? 0 : Math.Max(0, (int)Math.Floor(rawContracts));

                    // Apply Leverage and Cap
                    int leveraged = (int)(r.ContractsBase * _params.Leverage);
                    if (_params.MaxContracts > 0)
                        r.Contracts = Math.Clamp(leveraged, 0, _params.MaxContracts); // python clip(lower=0) -> allow 0 contracts
                    else
                        r.Contracts = Math.Max(0, leveraged); // changed from 1 to 0 to match Python allowing 0 contracts if not enough cap
                }
                else
                {
                    r.ContractsBase = 0;
                    r.Contracts = 0; // Wait, python's safe_atr replaces 0 with NaN -> NaN -> fillna(0) -> 0. Wait, then clip(lower=0). But wait, does 0 contracts allow trade?
                }
            }
        }
    }
}
