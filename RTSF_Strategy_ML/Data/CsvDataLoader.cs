using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using CsvHelper;
using CsvHelper.Configuration;
using RTSF_Strategy_ML.Core.Models;

namespace RTSF_Strategy_ML.Data
{
    public class CsvDataLoader
    {
        private readonly string _dataBasePath;

        public CsvDataLoader(string dataBasePath = @"G:\data2")
        {
            _dataBasePath = dataBasePath;
        }

        public List<Bar> LoadM1Bars(string ticker, DateTime startDate, DateTime endDate)
        {
            var bars = new List<Bar>();

            // Traverse the directory structure: YEAR / MONTH / DAY / TICKER / M1 / {data.csv or data.txt}
            if (!Directory.Exists(_dataBasePath))
            {
                throw new DirectoryNotFoundException($"Data base path '{_dataBasePath}' not found.");
            }

            var yearDirs = Directory.GetDirectories(_dataBasePath);
            
            foreach (var yearDir in yearDirs.OrderBy(d => d))
            {
                var yearName = Path.GetFileName(yearDir);
                if (!int.TryParse(yearName, out int year) || year < startDate.Year || year > endDate.Year)
                    continue;

                var monthDirs = Directory.GetDirectories(yearDir);
                foreach (var monthDir in monthDirs.OrderBy(d => d))
                {
                    var monthDirsName = Path.GetFileName(monthDir);
                    if (!int.TryParse(monthDirsName, out int month))
                        continue;

                    var dayDirs = Directory.GetDirectories(monthDir);
                    foreach (var dayDir in dayDirs.OrderBy(d => d))
                    {
                        var dayDirsName = Path.GetFileName(dayDir);
                        if (!int.TryParse(dayDirsName, out int day))
                            continue;

                        DateTime currentDate;
                        try
                        {
                            currentDate = new DateTime(year, month, day);
                        }
                        catch (ArgumentOutOfRangeException)
                        {
                            continue;
                        }

                        if (currentDate < startDate.Date || currentDate > endDate.Date)
                            continue;

                        // Look for ticker
                        string tickerM1Path = Path.Combine(dayDir, ticker, "M1");
                        if (!Directory.Exists(tickerM1Path))
                            continue;

                        // Look for data.csv or data.txt
                        string? dataFile = null;
                        if (File.Exists(Path.Combine(tickerM1Path, "data.txt")))
                            dataFile = Path.Combine(tickerM1Path, "data.txt");
                        else if (File.Exists(Path.Combine(tickerM1Path, "data.csv")))
                            dataFile = Path.Combine(tickerM1Path, "data.csv");

                        if (dataFile != null)
                        {
                            bars.AddRange(ParseFile(dataFile));
                        }
                    }
                }
            }

            // Remove duplicates by time and sort
            return bars
                .GroupBy(b => b.Time)
                .Select(g => g.First())
                .OrderBy(b => b.Time)
                .ToList();
        }

        private IEnumerable<Bar> ParseFile(string filePath)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = false,
                MissingFieldFound = null,
                BadDataFound = null
            };

            using var reader = new StreamReader(filePath);
            using var csv = new CsvReader(reader, config);

            while (csv.Read())
            {
                // M1_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]
                // date format: MM/dd/yy (e.g., 01/05/22)
                // time format: HHmmss (e.g., 100000) or Hmmss (e.g., 90000)

                var dateStr = csv.GetField<string>(0);
                var timeStr = csv.GetField<string>(1);

                if (string.IsNullOrWhiteSpace(dateStr) || string.IsNullOrWhiteSpace(timeStr))
                    continue;

                // Pad time string with leading zeros if necessary
                timeStr = timeStr.PadLeft(6, '0');

                var dateTimeStr = $"{dateStr} {timeStr}";

                if (!DateTime.TryParseExact(dateTimeStr, "MM/dd/yy HHmmss", CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime timestamp))
                {
                    // Fallback or skip
                    continue;
                }

                var open = csv.GetField<float>(2);
                var high = csv.GetField<float>(3);
                var low = csv.GetField<float>(4);
                var close = csv.GetField<float>(5);
                var volume = csv.GetField<long>(6);

                yield return new Bar(timestamp, open, high, low, close, volume);
            }
        }
    }
}
