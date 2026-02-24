using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommandLine;
using Microsoft.AspNetCore.SignalR.Client;
using RTSF_Strategy_ML.Core.Models;
using RTSF_Strategy_ML.Data;
using RTSF_Strategy_ML.Strategy;
using RTSF_Strategy_ML.Reporting;
using RTSF_Strategy_ML.ML;

namespace RTSF_Strategy_ML
{
    class Program
    {
        static async Task Main(string[] args)
        {
            await Parser.Default.ParseArguments<Options>(args)
                .WithParsedAsync(RunAsync);
        }

        private static async Task RunAsync(Options opts)
        {
            if (opts.Mode.Equals("backtest", StringComparison.OrdinalIgnoreCase))
            {
                RunBacktest(opts);
            }
            else if (opts.Mode.Equals("live", StringComparison.OrdinalIgnoreCase))
            {
                await RunLiveAsync(opts);
            }
            else
            {
                Console.WriteLine($"Unknown mode: {opts.Mode}. Use 'backtest' or 'live'.");
            }
        }

        private static void RunBacktest(Options opts)
        {
            Console.WriteLine(new string('=', 80));
            Console.WriteLine($"  BACKTEST MODE: {opts.Ticker} ({opts.StartDate} to {opts.EndDate})");
            Console.WriteLine($"  ML Filter: {opts.MlThreshold}, Leverage: {opts.Leverage}x");
            Console.WriteLine(new string('=', 80));

            var sw = Stopwatch.StartNew();
            var loader = new CsvDataLoader(opts.DataPath);
            
            DateTime startDate = DateTime.Parse(opts.StartDate);
            DateTime endDate = DateTime.Parse(opts.EndDate);

            // Load extra 6 months before start date to warm up indicators (D1 ADX and 210-min SMA)
            var warmupStart = startDate.AddMonths(-6);
            Console.WriteLine($"Loading M1 bars from {warmupStart:yyyy-MM-dd} (incl. warmup)...");
            var bars = loader.LoadM1Bars(opts.Ticker, warmupStart, endDate);
            Console.WriteLine($"Loaded {bars.Count:N0} M1 bars in {sw.ElapsedMilliseconds} ms.");

            if (bars.Count == 0)
            {
                Console.WriteLine("No data found.");
                return;
            }

            sw.Restart();
            Console.WriteLine("Aggregating timeframes and generating signals...");

            // Original parameters from block_e optimization (BLOCK_E_FULL_REPORT.md)
            var pLong = new StrategyParams { 
                Tf1Minutes = 150, Tf2Minutes = 60, Lookback = 95, Length = 280, Lookback2 = 85, Length2 = 185,
                Koeff1 = 0.9609f, Koeff2 = 1.0102f, MinS = 180, MaxS = 500, Mmcoff = 26,
                ExitDay = 0, SdelDay = 1, ExitWeek = 0,
                Direction = Core.Enums.TradeDirection.Long, Leverage = opts.Leverage, MaxContracts = 100
            };
            var pShort = new StrategyParams { 
                Tf1Minutes = 180, Tf2Minutes = 45, Lookback = 245, Length = 55, Lookback2 = 70, Length2 = 95,
                Koeff1 = 1.0000f, Koeff2 = 0.9458f, MinS = 330, MaxS = 650, Mmcoff = 17,
                ExitDay = 0, SdelDay = 1, ExitWeek = 0,
                Direction = Core.Enums.TradeDirection.Short, Leverage = opts.Leverage, MaxContracts = 40
            };

            var rowsLong = DataAggregator.PrepareStrategyData(bars, pLong.Tf1Minutes, pLong.Tf2Minutes);
            var rowsShort = DataAggregator.PrepareStrategyData(bars, pShort.Tf1Minutes, pShort.Tf2Minutes);

            var strategyLong = new MomentumTrendStrategy(pLong);
            strategyLong.GenerateSignals(rowsLong);

            var strategyShort = new MomentumTrendStrategy(pShort);
            strategyShort.GenerateSignals(rowsShort);

            // Discard warmup period rows before backtesting to avoid trading during warmup
            var targetRowsLong = rowsLong.Where(r => r.Time.Date >= startDate).ToList();
            var targetRowsShort = rowsShort.Where(r => r.Time.Date >= startDate).ToList();

            Console.WriteLine($"Simulating combined trades...");
            var trades = Backtester.SimulateCombinedTrades(targetRowsLong, targetRowsShort, pLong, pShort, Backtester.FlipMode.CloseLoss, 0.01f, tpAtrMult: opts.TpAtrMult, tpPct: opts.TpPct);
            Console.WriteLine($"Simulation complete. Raw trades: {trades.Count} in {sw.ElapsedMilliseconds} ms.");

            // ML Overlay Filter (skip when threshold >= 1.0 to disable)
            if (opts.MlThreshold < 1.0f && File.Exists(opts.MlModelPath))
            {
                sw.Restart();
                Console.WriteLine($"Applying ML Overlay (Threshold: {opts.MlThreshold})...");
                
                var featureBuilder = new FeatureBuilder(bars);
                using var scorer = new XGBoostScorer(opts.MlModelPath);

                var filteredTrades = new List<Trade>();

                foreach (var trade in trades)
                {
                    try
                    {
                        var features = featureBuilder.GetFeatures(trade.EntryTime, trade.Direction);
                        float prob = scorer.PredictProbability(features);
                        trade.MlScore = prob;

                        if (prob >= opts.MlThreshold)
                        {
                            filteredTrades.Add(trade);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"ML Error for trade at {trade.EntryTime}: {ex.Message}");
                    }
                }

                Console.WriteLine($"ML Filtered: kept {filteredTrades.Count} out of {trades.Count} trades ({filteredTrades.Count/(float)trades.Count:P1})");
                trades = filteredTrades;
            }
            else if (opts.MlThreshold >= 1.0f)
            {
                Console.WriteLine("ML Filter disabled (threshold >= 1.0).");
            }
            else
            {
                Console.WriteLine($"WARNING: ML Model not found at {opts.MlModelPath}. Running without ML overlay.");
            }

            // Reporting
            var metrics = KonkopAnalyzer.Analyze(trades, $"RTSF Backtest ({opts.StartDate} - {opts.EndDate})", 5000000);
            KonkopAnalyzer.PrintReport(metrics);
            KonkopAnalyzer.PrintYearlyGrid(trades, 5000000);
            KonkopAnalyzer.PrintMonthlyGrid(trades, 5000000);
            
            // Export trades to CSV
            string outCsv = $"block_e_csharp_trades_lev{opts.Leverage}.csv";
            ExportTradesToCsv(trades, outCsv);
            Console.WriteLine($"Exported {trades.Count} trades to {outCsv}");
        }

        private static void ExportTradesToCsv(System.Collections.Generic.List<Trade> trades, string path)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("Id,Direction,EntryTime,ExitTime,EntryPrice,ExitPrice,Contracts,Pnl1c,PnlNet,ExitReason,MlScore");
            foreach (var t in trades)
            {
                writer.WriteLine($"{t.Id},{t.Direction},{t.EntryTime:yyyy-MM-dd HH:mm:ss},{t.ExitTime:yyyy-MM-dd HH:mm:ss},{t.EntryPrice},{t.ExitPrice},{t.Contracts},{t.Pnl1c},{t.PnlNet},{t.ExitReason},{t.MlScore:F3}");
            }
        }

        private static async Task RunLiveAsync(Options opts)
        {
            Console.WriteLine(new string('=', 80));
            Console.WriteLine("  LIVE MODE");
            Console.WriteLine(new string('=', 80));

            var pLong = new StrategyParams { 
                Tf1Minutes = 150, Tf2Minutes = 60, Lookback = 95, Length = 280, Lookback2 = 85, Length2 = 185,
                Koeff1 = 0.9609f, Koeff2 = 1.0102f, MinS = 180, MaxS = 500, Mmcoff = 26,
                ExitDay = 0, SdelDay = 1, ExitWeek = 0,
                Direction = Core.Enums.TradeDirection.Long, Leverage = opts.Leverage, MaxContracts = 100
            };
            var pShort = new StrategyParams { 
                Tf1Minutes = 180, Tf2Minutes = 45, Lookback = 245, Length = 55, Lookback2 = 70, Length2 = 95,
                Koeff1 = 1.0000f, Koeff2 = 0.9458f, MinS = 330, MaxS = 650, Mmcoff = 17,
                ExitDay = 0, SdelDay = 1, ExitWeek = 0,
                Direction = Core.Enums.TradeDirection.Short, Leverage = opts.Leverage, MaxContracts = 40
            };

            XGBoostScorer? scorer = null;
            if (File.Exists(opts.MlModelPath))
            {
                Console.WriteLine("Loaded ML Model for live scoring.");
                scorer = new XGBoostScorer(opts.MlModelPath);
            }
            else
            {
                Console.WriteLine($"WARNING: ML Model not found at {opts.MlModelPath}. Running without ML overlay.");
            }

            if (!string.IsNullOrEmpty(opts.TinkoffToken))
            {
                await RunLiveTinkoffAsync(opts, pLong, pShort, scorer);
            }
            else if (!string.IsNullOrEmpty(opts.SignalRUrl))
            {
                await RunLiveSignalRAsync(opts, pLong, pShort, scorer);
            }
            else
            {
                Console.WriteLine("Error: --tinkoff_token or --signalr URL is required for live mode.");
            }
        }

        private static async Task RunLiveTinkoffAsync(Options opts, StrategyParams pLong, StrategyParams pShort, XGBoostScorer? scorer)
        {
            Console.WriteLine($"Connecting to Tinkoff Invest API...");
            var provider = new TinkoffDataProvider(opts.TinkoffToken);

            if (!await provider.FindFutureAsync(opts.FuturePrefix))
            {
                Console.WriteLine("Failed to find futures instrument. Exiting.");
                return;
            }

            // Load warmup data from local CSV + recent Tinkoff candles
            var bars = new List<Bar>();
            DateTime now = DateTime.Now;

            if (!string.IsNullOrEmpty(opts.DataPath) && Directory.Exists(opts.DataPath))
            {
                Console.WriteLine("Loading warmup bars from local CSV...");
                var loader = new CsvDataLoader(opts.DataPath);
                bars = loader.LoadM1Bars(opts.Ticker, now.AddMonths(-6), now);
                Console.WriteLine($"  Local CSV: {bars.Count:N0} M1 bars");
            }

            // Supplement with recent Tinkoff candles (last 5 days)
            Console.WriteLine("Loading recent candles from Tinkoff API...");
            var recentBars = await provider.LoadHistoricalCandlesAsync(
                now.AddDays(-5).ToUniversalTime(), now.ToUniversalTime());
            Console.WriteLine($"  Tinkoff API: {recentBars.Count:N0} M1 bars");

            // Merge: keep local bars that are older than Tinkoff bars, then append Tinkoff
            if (recentBars.Count > 0)
            {
                var tinkoffStart = recentBars.Min(b => b.Time);
                bars = bars.Where(b => b.Time < tinkoffStart).ToList();
                bars.AddRange(recentBars);
                bars = bars.OrderBy(b => b.Time).ToList();
            }
            Console.WriteLine($"  Total warmup: {bars.Count:N0} bars ({bars.FirstOrDefault()?.Time:yyyy-MM-dd HH:mm} to {bars.LastOrDefault()?.Time:yyyy-MM-dd HH:mm})");

            var stratLong = new MomentumTrendStrategy(pLong);
            var stratShort = new MomentumTrendStrategy(pShort);
            FeatureBuilder? featureBuilder = bars.Count > 0 ? new FeatureBuilder(bars) : null;

            int barCount = 0;
            var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

            Console.WriteLine(new string('=', 80));
            Console.WriteLine($"  Streaming M1 candles for {provider.Ticker}. Press Ctrl+C to stop.");
            Console.WriteLine(new string('=', 80));

            await provider.StreamCandlesAsync(newBar =>
            {
                barCount++;
                bars.Add(newBar);

                // Recalculate signals on each new bar
                var rowsL = DataAggregator.PrepareStrategyData(bars, pLong.Tf1Minutes, pLong.Tf2Minutes);
                stratLong.GenerateSignals(rowsL);
                var lastL = rowsL.LastOrDefault();

                var rowsS = DataAggregator.PrepareStrategyData(bars, pShort.Tf1Minutes, pShort.Tf2Minutes);
                stratShort.GenerateSignals(rowsS);
                var lastS = rowsS.LastOrDefault();

                string signalStr = "";
                if (lastL != null && lastL.EntrySignal)
                {
                    float mlProb = 1f;
                    if (scorer != null && featureBuilder != null)
                    {
                        try
                        {
                            var features = featureBuilder.GetFeatures(lastL.Time, Core.Enums.TradeDirection.Long);
                            mlProb = scorer.PredictProbability(features);
                        }
                        catch { }
                    }
                    signalStr = mlProb >= opts.MlThreshold
                        ? $" >>> LONG ENTRY (ML={mlProb:F3}) <<<"
                        : $" [LONG filtered by ML={mlProb:F3}]";
                }
                if (lastS != null && lastS.EntrySignal)
                {
                    float mlProb = 1f;
                    if (scorer != null && featureBuilder != null)
                    {
                        try
                        {
                            var features = featureBuilder.GetFeatures(lastS.Time, Core.Enums.TradeDirection.Short);
                            mlProb = scorer.PredictProbability(features);
                        }
                        catch { }
                    }
                    signalStr += mlProb >= opts.MlThreshold
                        ? $" >>> SHORT ENTRY (ML={mlProb:F3}) <<<"
                        : $" [SHORT filtered by ML={mlProb:F3}]";
                }

                string exitStr = "";
                if (lastL != null && lastL.ExitSignal) exitStr += " [LONG EXIT]";
                if (lastS != null && lastS.ExitSignal) exitStr += " [SHORT EXIT]";

                Console.WriteLine($"[{newBar.Time:HH:mm:ss}] O={newBar.Open:F0} H={newBar.High:F0} L={newBar.Low:F0} C={newBar.Close:F0} V={newBar.Volume}{signalStr}{exitStr}");

            }, cts.Token);

            Console.WriteLine($"\nStream ended. Received {barCount} bars.");
            scorer?.Dispose();
        }

        private static async Task RunLiveSignalRAsync(Options opts, StrategyParams pLong, StrategyParams pShort, XGBoostScorer? scorer)
        {
            Console.WriteLine($"Connecting to SignalR Hub: {opts.SignalRUrl}...");
            
            var loader = new CsvDataLoader(opts.DataPath);
            DateTime endDate = DateTime.Now;
            DateTime warmupStart = endDate.AddMonths(-3);
            
            Console.WriteLine($"Warming up indicators from local CSV data ({warmupStart:yyyy-MM-dd} to {endDate:yyyy-MM-dd})...");
            var bars = loader.LoadM1Bars(opts.Ticker, warmupStart, endDate);
            Console.WriteLine($"Loaded {bars.Count:N0} M1 bars for warmup.");

            var connection = new HubConnectionBuilder()
                .WithUrl(opts.SignalRUrl)
                .WithAutomaticReconnect()
                .Build();

            connection.On<string, float, float, float, float, long>("NewBar", (timeStr, open, high, low, close, volume) =>
            {
                if (DateTime.TryParse(timeStr, out DateTime barTime))
                {
                    var newBar = new Bar(barTime, open, high, low, close, volume);
                    Console.WriteLine($"[New Bar] {newBar}");
                    bars.Add(newBar);
                }
            });

            try
            {
                await connection.StartAsync();
                Console.WriteLine("Connected to SignalR! Waiting for live bars... (Press Ctrl+C to exit)");
                await Task.Delay(-1);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error connecting to SignalR: {ex.Message}");
            }
            finally
            {
                scorer?.Dispose();
            }
        }
    }
}
