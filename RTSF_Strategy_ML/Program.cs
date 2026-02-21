using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
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

            // Get base config from our previously generated "v3_joint_best" parameters
            var pLong = new StrategyParams { 
                Tf1Minutes = 210, Tf2Minutes = 90, Lookback = 165, Length = 180, Lookback2 = 30, Length2 = 103,
                Koeff1 = 1.0f, Koeff2 = 1.005f, MinS = 290, MaxS = 480, Direction = Core.Enums.TradeDirection.Long,
                Leverage = opts.Leverage
            };
            var pShort = new StrategyParams { 
                Tf1Minutes = 210, Tf2Minutes = 90, Lookback = 165, Length = 180, Lookback2 = 30, Length2 = 103,
                Koeff1 = 1.0f, Koeff2 = 1.005f, MinS = 290, MaxS = 480, Direction = Core.Enums.TradeDirection.Short,
                Leverage = opts.Leverage
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
            var trades = Backtester.SimulateCombinedTrades(targetRowsLong, targetRowsShort, pLong, pShort, Backtester.FlipMode.CloseLoss, 0.01f);
            Console.WriteLine($"Simulation complete. Raw trades: {trades.Count} in {sw.ElapsedMilliseconds} ms.");

            // ML Overlay Filter
            if (File.Exists(opts.MlModelPath))
            {
                sw.Restart();
                Console.WriteLine($"Applying ML Overlay (Threshold: {opts.MlThreshold})...");
                
                var featureBuilder = new FeatureBuilder(bars); // use full bars for feature history
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
            Console.WriteLine("  LIVE TRADING MODE (SignalR)");
            Console.WriteLine(new string('=', 80));

            if (string.IsNullOrEmpty(opts.SignalRUrl))
            {
                Console.WriteLine("Error: --signalr URL is required for live mode.");
                return;
            }

            Console.WriteLine($"Connecting to SignalR Hub: {opts.SignalRUrl}...");
            
            // 1. Инициализация состояния: загрузка "хвоста" исторических M1 баров (за последние 3 месяца)
            var loader = new CsvDataLoader(opts.DataPath);
            DateTime endDate = DateTime.Now;
            DateTime warmupStart = endDate.AddMonths(-3);
            
            Console.WriteLine($"Warming up indicators from local CSV data ({warmupStart:yyyy-MM-dd} to {endDate:yyyy-MM-dd})...");
            var bars = loader.LoadM1Bars(opts.Ticker, warmupStart, endDate);
            Console.WriteLine($"Loaded {bars.Count:N0} M1 bars for warmup.");

            // Get base config 
            var pLong = new StrategyParams { 
                Tf1Minutes = 210, Tf2Minutes = 90, Lookback = 165, Length = 180, Lookback2 = 30, Length2 = 103,
                Koeff1 = 1.0f, Koeff2 = 1.005f, MinS = 290, MaxS = 480, Direction = Core.Enums.TradeDirection.Long,
                Leverage = opts.Leverage
            };
            var pShort = new StrategyParams { 
                Tf1Minutes = 210, Tf2Minutes = 90, Lookback = 165, Length = 180, Lookback2 = 30, Length2 = 103,
                Koeff1 = 1.0f, Koeff2 = 1.005f, MinS = 290, MaxS = 480, Direction = Core.Enums.TradeDirection.Short,
                Leverage = opts.Leverage
            };

            // ML Overlay
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

            // 2. Подключение к SignalR хабу
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
                    
                    // Add new bar to history
                    bars.Add(newBar);

                    // Re-calculate last row strategy signals (In real production, you'd only update the tip, not the whole array, but for this demo we'll just run the tip logic)
                    // TODO: Implement incremental update logic instead of full recalculation
                    
                    // Example check:
                    // var rowsLong = DataAggregator.PrepareStrategyData(bars, pLong.Tf1Minutes, pLong.Tf2Minutes);
                    // var strategyLong = new MomentumTrendStrategy(pLong);
                    // strategyLong.GenerateSignals(rowsLong);
                    // var lastRow = rowsLong.Last();
                    
                    // if (lastRow.EntrySignal) {
                    //      var featureBuilder = new FeatureBuilder(bars);
                    //      var features = featureBuilder.GetFeatures(lastRow.Time, TradeDirection.Long);
                    //      if (scorer != null) {
                    //          float prob = scorer.PredictProbability(features);
                    //          if (prob >= opts.MlThreshold) {
                    //              Console.WriteLine(">>> LIVE TRADE SIGNAL: GO LONG! Prob: " + prob);
                    //          }
                    //      }
                    // }
                }
            });

            try
            {
                await connection.StartAsync();
                Console.WriteLine("Connected to SignalR! Waiting for live bars... (Press Ctrl+C to exit)");
                
                // Block indefinitely
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
