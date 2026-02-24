using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Google.Protobuf.WellKnownTypes;
using Tinkoff.InvestApi;
using Tinkoff.InvestApi.V1;
using RTSF_Strategy_ML.Core.Models;

namespace RTSF_Strategy_ML.Data
{
    public class TinkoffDataProvider : IDisposable
    {
        private readonly InvestApiClient _client;
        private string? _instrumentUid;
        private string? _figi;
        private string _ticker = "";
        private float _minPriceIncrement;

        public string InstrumentUid => _instrumentUid ?? "";
        public string Figi => _figi ?? "";
        public string Ticker => _ticker;

        public TinkoffDataProvider(string token)
        {
            _client = InvestApiClientFactory.Create(token);
        }

        public async Task<bool> FindFutureAsync(string tickerPrefix)
        {
            var futures = await _client.Instruments.FuturesAsync(new InstrumentsRequest
            {
                InstrumentStatus = InstrumentStatus.All
            });

            Console.WriteLine($"Total futures from API: {futures.Instruments.Count}");

            var allMatching = futures.Instruments
                .Where(f => f.Ticker.IndexOf(tickerPrefix, StringComparison.OrdinalIgnoreCase) >= 0
                            || f.Name.IndexOf(tickerPrefix, StringComparison.OrdinalIgnoreCase) >= 0
                            || f.BasicAsset.IndexOf(tickerPrefix, StringComparison.OrdinalIgnoreCase) >= 0)
                .OrderBy(f => f.LastTradeDate)
                .ToList();

            var found = allMatching
                .Where(f => f.LastTradeDate.ToDateTime() > DateTime.UtcNow)
                .FirstOrDefault();

            if (found == null)
            {
                Console.WriteLine($"No active futures found for '{tickerPrefix}'. Matches ({allMatching.Count}):");
                foreach (var f in allMatching.TakeLast(15))
                    Console.WriteLine($"  {f.Ticker} ({f.Name}) FIGI={f.Figi} UID={f.Uid} Trade={f.ApiTradeAvailableFlag} Expires={f.LastTradeDate} Asset={f.BasicAsset}");

                if (allMatching.Count == 0)
                {
                    Console.WriteLine("Listing first 20 futures for reference:");
                    foreach (var f in futures.Instruments.Take(20))
                        Console.WriteLine($"  {f.Ticker} ({f.Name}) Asset={f.BasicAsset}");
                }
                return false;
            }

            _figi = found.Figi;
            _instrumentUid = found.Uid;
            _ticker = found.Ticker;
            _minPriceIncrement = QuotationToFloat(found.MinPriceIncrement);

            Console.WriteLine($"Found future: {found.Ticker} ({found.Name})");
            Console.WriteLine($"  FIGI={found.Figi}, UID={found.Uid}");
            Console.WriteLine($"  Lot={found.Lot}, MinPriceIncrement={_minPriceIncrement}");
            Console.WriteLine($"  Expires: {found.LastTradeDate}");
            return true;
        }

        public async Task<List<Bar>> LoadHistoricalCandlesAsync(DateTime from, DateTime to, CandleInterval interval = CandleInterval._1Min)
        {
            if (string.IsNullOrEmpty(_instrumentUid))
                throw new InvalidOperationException("Call FindFutureAsync first");

            var bars = new List<Bar>();
            var current = from;

            while (current < to)
            {
                var chunkEnd = current.AddDays(1);
                if (chunkEnd > to) chunkEnd = to;

                try
                {
                    var response = await _client.MarketData.GetCandlesAsync(new GetCandlesRequest
                    {
                        InstrumentId = _instrumentUid,
                        From = Timestamp.FromDateTime(DateTime.SpecifyKind(current, DateTimeKind.Utc)),
                        To = Timestamp.FromDateTime(DateTime.SpecifyKind(chunkEnd, DateTimeKind.Utc)),
                        Interval = interval,
                    });

                    foreach (var c in response.Candles)
                    {
                        if (!c.IsComplete) continue;
                        bars.Add(new Bar(
                            c.Time.ToDateTime().ToLocalTime(),
                            QuotationToFloat(c.Open),
                            QuotationToFloat(c.High),
                            QuotationToFloat(c.Low),
                            QuotationToFloat(c.Close),
                            c.Volume
                        ));
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Warning: candle fetch error for {current:yyyy-MM-dd}: {ex.Message}");
                }

                current = chunkEnd;
                await Task.Delay(200);
            }

            return bars.OrderBy(b => b.Time).ToList();
        }

        public async Task StreamCandlesAsync(
            Action<Bar> onNewBar,
            CancellationToken ct,
            SubscriptionInterval subInterval = SubscriptionInterval.OneMinute)
        {
            if (string.IsNullOrEmpty(_figi))
                throw new InvalidOperationException("Call FindFutureAsync first");

            Console.WriteLine($"Starting candle stream for {_ticker} (FIGI={_figi}, interval={subInterval})...");
            Console.WriteLine($"  WaitingClose=true for closed candles + Trades for tick display...");

            while (!ct.IsCancellationRequested)
            {
                try
                {
                    var deadline = DateTime.UtcNow.AddHours(24);
                    using var stream = _client.MarketDataStream.MarketDataStream(
                        deadline: deadline,
                        cancellationToken: ct);

                    // Closed M1 candles for strategy signals
                    await stream.RequestStream.WriteAsync(new MarketDataRequest
                    {
                        SubscribeCandlesRequest = new SubscribeCandlesRequest
                        {
                            SubscriptionAction = SubscriptionAction.Subscribe,
                            WaitingClose = true,
                            Instruments =
                            {
                                new CandleInstrument
                                {
                                    InstrumentId = _figi,
                                    Interval = subInterval,
                                }
                            },
                        }
                    });

                    // Trades stream for tick-by-tick M1 bar building (fallback + display)
                    await stream.RequestStream.WriteAsync(new MarketDataRequest
                    {
                        SubscribeTradesRequest = new SubscribeTradesRequest
                        {
                            SubscriptionAction = SubscriptionAction.Subscribe,
                            Instruments =
                            {
                                new TradeInstrument { InstrumentId = _figi }
                            }
                        }
                    });

                    int tickCount = 0;

                    while (await stream.ResponseStream.MoveNext(ct))
                    {
                        var response = stream.ResponseStream.Current;

                        if (response.Candle != null)
                        {
                            var c = response.Candle;
                            var bar = new Bar(
                                c.Time.ToDateTime().ToLocalTime(),
                                QuotationToFloat(c.Open),
                                QuotationToFloat(c.High),
                                QuotationToFloat(c.Low),
                                QuotationToFloat(c.Close),
                                c.Volume
                            );
                            onNewBar(bar);
                        }

                        if (response.Trade != null)
                        {
                            var t = response.Trade;
                            float price = QuotationToFloat(t.Price);
                            var time = t.Time.ToDateTime().ToLocalTime();

                            tickCount++;
                            if (tickCount % 50 == 0 || tickCount < 5)
                                Console.Write($"\r  [{time:HH:mm:ss}] #{tickCount} P={price:F0} V={t.Quantity}       ");
                        }

                        if (response.SubscribeCandlesResponse != null)
                        {
                            foreach (var sub in response.SubscribeCandlesResponse.CandlesSubscriptions)
                                Console.WriteLine($"  Candle sub: {sub.InstrumentUid} -> {sub.SubscriptionStatus}");
                        }

                        if (response.SubscribeTradesResponse != null)
                        {
                            foreach (var sub in response.SubscribeTradesResponse.TradeSubscriptions)
                                Console.WriteLine($"  Trades sub: {sub.InstrumentUid} -> {sub.SubscriptionStatus}");
                        }

                        if (response.Ping != null)
                        {
                            Console.Write($"\r  [ping {DateTime.Now:HH:mm:ss}]                              ");
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\n  Stream error: {ex.Message}. Reconnecting in 5s...");
                    await Task.Delay(5000, ct);
                }
            }
        }

        private static float QuotationToFloat(Quotation q)
        {
            return (float)(q.Units + q.Nano / 1_000_000_000.0);
        }

        public void Dispose() { }
    }

    /// <summary>
    /// Builds M1 bars from tick-level trade data as a fallback when candle stream is unavailable.
    /// </summary>
    public class M1CandleBuilder
    {
        private DateTime _currentMinute;
        private float _open, _high, _low, _close;
        private long _volume;
        private bool _hasData;

        public Bar? OnTick(DateTime time, float price, long qty = 1)
        {
            var minute = new DateTime(time.Year, time.Month, time.Day, time.Hour, time.Minute, 0);
            Bar? completed = null;

            if (_hasData && minute > _currentMinute)
            {
                completed = new Bar(_currentMinute, _open, _high, _low, _close, _volume);
                _hasData = false;
            }

            if (!_hasData)
            {
                _currentMinute = minute;
                _open = _high = _low = _close = price;
                _volume = qty;
                _hasData = true;
            }
            else
            {
                if (price > _high) _high = price;
                if (price < _low) _low = price;
                _close = price;
                _volume += qty;
            }

            return completed;
        }
    }
}
