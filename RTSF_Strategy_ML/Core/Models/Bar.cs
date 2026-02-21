using System;

namespace RTSF_Strategy_ML.Core.Models
{
    public class Bar
    {
        public DateTime Time { get; set; }
        public float Open { get; set; }
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
        public long Volume { get; set; }
        
        public Bar() { }

        public Bar(DateTime time, float open, float high, float low, float close, long volume)
        {
            Time = time;
            Open = open;
            High = high;
            Low = low;
            Close = close;
            Volume = volume;
        }

        public override string ToString()
        {
            return $"{Time:yyyy-MM-dd HH:mm:ss} | O: {Open} H: {High} L: {Low} C: {Close} V: {Volume}";
        }
    }
}
