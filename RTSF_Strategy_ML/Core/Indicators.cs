using System;
using System.Collections.Generic;
using System.Linq;
using RTSF_Strategy_ML.Core.Models;

namespace RTSF_Strategy_ML.Core
{
    public static class Indicators
    {
        /// <summary>
        /// Simple Moving Average (SMA) with min_periods=1 behavior (matches pandas rolling(period, min_periods=1).mean()).
        /// Returns partial averages from bar 0 when fewer than <paramref name="period"/> values are available.
        /// </summary>
        public static float[] Sma(float[] series, int period)
        {
            var result = new float[series.Length];
            if (series.Length == 0) return result;

            for (int i = 0; i < series.Length; i++)
            {
                int windowStart = Math.Max(0, i - period + 1);
                float sum = 0;
                int validCount = 0;
                for (int j = windowStart; j <= i; j++)
                {
                    float val = series[j];
                    if (!float.IsNaN(val))
                    {
                        sum += val;
                        validCount++;
                    }
                }

                result[i] = validCount > 0 ? sum / validCount : float.NaN;
            }
            return result;
        }

        /// <summary>
        /// Computes True Range (TR) for a list of bars.
        /// </summary>
        public static float[] TrueRange(List<Bar> bars)
        {
            var tr = new float[bars.Count];
            if (bars.Count == 0) return tr;

            tr[0] = bars[0].High - bars[0].Low;
            for (int i = 1; i < bars.Count; i++)
            {
                var prevClose = bars[i - 1].Close;
                var h = bars[i].High;
                var l = bars[i].Low;
                
                var tr1 = h - l;
                var tr2 = Math.Abs(h - prevClose);
                var tr3 = Math.Abs(l - prevClose);
                
                tr[i] = Math.Max(tr1, Math.Max(tr2, tr3));
            }
            return tr;
        }

        /// <summary>
        /// Average True Range (ATR) calculated using Simple Moving Average (as in momentum_trend.py).
        /// </summary>
        public static float[] Atr(List<Bar> bars, int period)
        {
            var tr = TrueRange(bars);
            var atr = new float[tr.Length];
            
            float sum = 0;
            for (int i = 0; i < tr.Length; i++)
            {
                sum += tr[i];
                if (i < period - 1)
                {
                    // Pandas rolling with min_periods=1 returns average of available items
                    atr[i] = sum / (i + 1);
                }
                else
                {
                    if (i >= period)
                    {
                        sum -= tr[i - period];
                    }
                    atr[i] = sum / period;
                }
            }
            return atr;
        }

        /// <summary>
        /// Computes the TrendScore for the given series and lookback.
        /// TrendScore = Sum over i in 1..10 of Sign(close[t] - close[t - (lookback + i)])
        /// </summary>
        public static float[] TrendScore(float[] close, int lookback)
        {
            var score = new float[close.Length];
            for (int i = 0; i < close.Length; i++)
            {
                float sumScore = 0;
                for (int j = 1; j <= 10; j++)
                {
                    int shift = lookback + j;
                    if (i >= shift)
                    {
                        float valNow = close[i];
                        float valShifted = close[i - shift];
                        if (!float.IsNaN(valNow) && !float.IsNaN(valShifted))
                        {
                            float diff = valNow - valShifted;
                            // Math.Sign expects float to behave like Python's np.sign
                            if (diff > 0) sumScore += 1;
                            else if (diff < 0) sumScore -= 1;
                        }
                    }
                }
                score[i] = sumScore;
            }
            return score;
        }

        /// <summary>
        /// Average Directional Index (ADX) using Wilder's Smoothing (alpha = 1 / period).
        /// </summary>
        public static float[] Adx(List<Bar> bars, int period = 14)
        {
            int n = bars.Count;
            var adx = new float[n];
            if (n == 0) return adx;

            var tr = TrueRange(bars);
            var plusDm = new float[n];
            var minusDm = new float[n];

            for (int i = 1; i < n; i++)
            {
                var upMove = bars[i].High - bars[i - 1].High;
                var downMove = bars[i - 1].Low - bars[i].Low;

                if (upMove > downMove && upMove > 0)
                    plusDm[i] = upMove;
                else
                    plusDm[i] = 0;

                if (downMove > upMove && downMove > 0)
                    minusDm[i] = downMove;
                else
                    minusDm[i] = 0;
            }

            // Wilder's smoothing function (exponential moving average with alpha = 1 / period)
            float alpha = 1.0f / period;
            
            var smoothTr = new float[n];
            var smoothPlusDm = new float[n];
            var smoothMinusDm = new float[n];

            if (n > 0)
            {
                smoothTr[0] = tr[0];
                smoothPlusDm[0] = plusDm[0];
                smoothMinusDm[0] = minusDm[0];
            }

            for (int i = 1; i < n; i++)
            {
                smoothTr[i] = alpha * tr[i] + (1 - alpha) * smoothTr[i - 1];
                smoothPlusDm[i] = alpha * plusDm[i] + (1 - alpha) * smoothPlusDm[i - 1];
                smoothMinusDm[i] = alpha * minusDm[i] + (1 - alpha) * smoothMinusDm[i - 1];
            }

            var dx = new float[n];
            for (int i = 0; i < n; i++)
            {
                float atrVal = smoothTr[i];
                float plusDi = atrVal == 0 ? 0 : 100 * smoothPlusDm[i] / atrVal;
                float minusDi = atrVal == 0 ? 0 : 100 * smoothMinusDm[i] / atrVal;

                float sumDi = plusDi + minusDi;
                dx[i] = sumDi == 0 ? 0 : 100 * Math.Abs(plusDi - minusDi) / (sumDi + 1e-9f);
            }

            if (n > 0)
            {
                adx[0] = dx[0];
            }

            for (int i = 1; i < n; i++)
            {
                adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1];
            }

            return adx;
        }

        /// <summary>
        /// Exponential Moving Average (EMA).
        /// Used primarily in feature extraction for ML.
        /// </summary>
        public static float[] Ema(float[] series, int span)
        {
            var result = new float[series.Length];
            if (series.Length == 0) return result;

            float alpha = 2.0f / (span + 1);
            result[0] = series[0];
            for (int i = 1; i < series.Length; i++)
            {
                result[i] = alpha * series[i] + (1 - alpha) * result[i - 1];
            }
            return result;
        }
    }
}
