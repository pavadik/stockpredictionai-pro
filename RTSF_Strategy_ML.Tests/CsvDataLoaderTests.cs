using System;
using System.IO;
using System.Linq;
using RTSF_Strategy_ML.Data;
using Xunit;

namespace RTSF_Strategy_ML.Tests
{
    public class CsvDataLoaderTests : IDisposable
    {
        private readonly string _tempPath;

        public CsvDataLoaderTests()
        {
            _tempPath = Path.Combine(Path.GetTempPath(), "RTSF_Test_Data_" + Guid.NewGuid());
            Directory.CreateDirectory(_tempPath);
        }

        public void Dispose()
        {
            if (Directory.Exists(_tempPath))
            {
                Directory.Delete(_tempPath, true);
            }
        }

        [Fact]
        public void LoadM1Bars_ParsesCorrectly()
        {
            // Setup directory structure for 2016/01/04/RTSF/M1/data.csv
            var m1Dir = Path.Combine(_tempPath, "2016", "01", "04", "RTSF", "M1");
            Directory.CreateDirectory(m1Dir);
            
            // Format: "date", "time", "open", "high", "low", "close", "volume"
            // MM/dd/yy HHmmss
            var csvContent = @"01/04/16,100000,75390,76320,75250,76210,7287
01/04/16,100100,76220,76270,76140,76140,2635
01/04/16,90000,75000,75000,75000,75000,1000"; // Added 9:00:00 to test padding

            File.WriteAllText(Path.Combine(m1Dir, "data.csv"), csvContent);

            var loader = new CsvDataLoader(_tempPath);
            var bars = loader.LoadM1Bars("RTSF", new DateTime(2016, 1, 1), new DateTime(2016, 1, 31));

            Assert.Equal(3, bars.Count);

            // Should be sorted by time
            Assert.Equal(new DateTime(2016, 1, 4, 9, 0, 0), bars[0].Time);
            Assert.Equal(75000f, bars[0].Open);

            Assert.Equal(new DateTime(2016, 1, 4, 10, 0, 0), bars[1].Time);
            Assert.Equal(75390f, bars[1].Open);
            Assert.Equal(76320f, bars[1].High);
            Assert.Equal(75250f, bars[1].Low);
            Assert.Equal(76210f, bars[1].Close);
            Assert.Equal(7287L, bars[1].Volume);

            Assert.Equal(new DateTime(2016, 1, 4, 10, 1, 0), bars[2].Time);
            Assert.Equal(76220f, bars[2].Open);
        }
    }
}