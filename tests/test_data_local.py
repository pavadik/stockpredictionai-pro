"""Tests for local MOEX data loader (data_local.py).

Uses synthetic fixture data in tests/fixtures/moex_sample/ to test offline,
without requiring access to G:\\data2.

Fixture layout:
    moex_sample/2020/3/{16,17,18}/SBRF/M1/data.csv   (400 bars/day)
    moex_sample/2020/3/{16,17}/GMKN/M1/data.csv       (400 bars/day)
    moex_sample/2020/3/{16,17,18}/LKOH/M1/data.txt    (400 bars/day, txt!)
    moex_sample/2020/3/{16,17,18}/SBRF/ticks/data.txt  (500 ticks/day)
    moex_sample/2020/3/{16,17,18}/LKOH/ticks/data.txt  (800 ticks/day)
"""
import os
import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    Config, parse_timeframe, timeframe_mult, _wf_defaults_for,
)
from src.data_local import (
    load_m1_bars, load_ticks, aggregate_timeframe, ticks_to_bars,
    load_bars, scan_available_tickers, build_local_panel,
    _find_data_file,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "moex_sample")


# ======================================================================
# parse_timeframe / timeframe_mult
# ======================================================================

class TestParseTimeframe:
    def test_m1(self):
        info = parse_timeframe("M1")
        assert info["unit"] == "M" and info["n"] == 1 and info["minutes"] == 1

    def test_m5(self):
        info = parse_timeframe("M5")
        assert info["unit"] == "M" and info["n"] == 5 and info["minutes"] == 5
        assert info["resample_rule"] == "5min"

    def test_m14(self):
        info = parse_timeframe("M14")
        assert info["minutes"] == 14
        assert info["resample_rule"] == "14min"

    def test_m30(self):
        info = parse_timeframe("M30")
        assert info["minutes"] == 30

    def test_h1(self):
        info = parse_timeframe("H1")
        assert info["unit"] == "H" and info["n"] == 1 and info["minutes"] == 60

    def test_h4(self):
        info = parse_timeframe("H4")
        assert info["minutes"] == 240
        assert info["resample_rule"] == "4h"

    def test_d1(self):
        info = parse_timeframe("D1")
        assert info["unit"] == "D" and info["n"] == 1

    def test_tick(self):
        info = parse_timeframe("tick")
        assert info["unit"] == "tick"

    def test_case_insensitive(self):
        info = parse_timeframe("m5")
        assert info["unit"] == "M" and info["n"] == 5

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_timeframe("X5")

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            parse_timeframe("M0")


class TestTimeframeMult:
    def test_d1(self):
        assert timeframe_mult("D1") == 1

    def test_h1(self):
        assert timeframe_mult("H1") == 390 // 60  # == 6

    def test_m1(self):
        assert timeframe_mult("M1") == 390

    def test_m5(self):
        assert timeframe_mult("M5") == 390 // 5  # == 78

    def test_m3(self):
        assert timeframe_mult("M3") == 390 // 3  # == 130

    def test_m14(self):
        assert timeframe_mult("M14") == 390 // 14  # == 27

    def test_m30(self):
        assert timeframe_mult("M30") == 390 // 30  # == 13

    def test_h4(self):
        assert timeframe_mult("H4") == 390 // 240  # == 1

    def test_tick(self):
        assert timeframe_mult("tick") == 1


class TestWfDefaults:
    def test_d1(self):
        wf = _wf_defaults_for("D1")
        assert wf["step"] == 120
        assert wf["min_train"] == 500

    def test_m5(self):
        wf = _wf_defaults_for("M5")
        mult = timeframe_mult("M5")
        assert wf["step"] == 120 * mult
        assert wf["min_train"] == 500 * mult


# ======================================================================
# File detection
# ======================================================================

class TestFindDataFile:
    def test_finds_csv(self):
        m1_dir = os.path.join(FIXTURES_DIR, "2020", "3", "16", "SBRF", "M1")
        path = _find_data_file(m1_dir)
        assert path is not None
        assert os.path.basename(path) == "data.csv"

    def test_finds_txt(self):
        m1_dir = os.path.join(FIXTURES_DIR, "2020", "3", "16", "LKOH", "M1")
        path = _find_data_file(m1_dir)
        assert path is not None
        assert os.path.basename(path) == "data.txt"

    def test_returns_none_for_empty(self, tmp_path):
        assert _find_data_file(str(tmp_path)) is None


# ======================================================================
# M1 loader
# ======================================================================

class TestLoadM1Bars:
    def test_columns(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-18")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_datetime_index(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-18")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_date_filter_single_day(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16")
        unique_dates = set(df.index.date)
        assert len(unique_dates) == 1

    def test_date_filter_range(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-18")
        assert len(df) > 400

    def test_two_digit_year_parsed_correctly(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16")
        assert df.index[0].year == 2020

    def test_no_nans(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-18")
        assert df.isna().sum().sum() == 0

    def test_no_duplicate_timestamps(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-18")
        assert not df.index.duplicated().any()

    def test_sorted_index(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-18")
        assert df.index.is_monotonic_increasing

    def test_missing_ticker_raises(self):
        with pytest.raises(RuntimeError, match="No M1 data found"):
            load_m1_bars(FIXTURES_DIR, "NONEXISTENT", "2020-03-16", "2020-03-18")

    def test_float32_dtypes(self):
        df = load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16")
        for col in ["open", "high", "low", "close"]:
            assert df[col].dtype == np.float32

    def test_loads_data_txt(self):
        """LKOH has data.txt (not data.csv) -- verify it loads."""
        df = load_m1_bars(FIXTURES_DIR, "LKOH", "2020-03-16", "2020-03-18")
        assert len(df) > 0
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# ======================================================================
# Tick loader
# ======================================================================

class TestLoadTicks:
    def test_columns(self):
        df = load_ticks(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16")
        assert list(df.columns) == ["price", "volume"]

    def test_datetime_index(self):
        df = load_ticks(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_row_count(self):
        df = load_ticks(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16")
        assert len(df) == 500

    def test_missing_raises(self):
        with pytest.raises(RuntimeError, match="No tick data found"):
            load_ticks(FIXTURES_DIR, "NONEXISTENT", "2020-03-16", "2020-03-16")

    def test_multi_day(self):
        df = load_ticks(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-18")
        assert len(df) >= 1500  # 3 days x 500

    def test_lkoh_ticks(self):
        df = load_ticks(FIXTURES_DIR, "LKOH", "2020-03-16", "2020-03-18")
        assert len(df) >= 2400  # 3 days x 800


# ======================================================================
# M1 aggregation (arbitrary timeframes)
# ======================================================================

class TestAggregateTimeframe:
    @pytest.fixture
    def m1_data(self):
        return load_m1_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16")

    def test_m1_noop(self, m1_data):
        result = aggregate_timeframe(m1_data, "M1")
        assert len(result) == len(m1_data)

    def test_m3_reduces_rows(self, m1_data):
        result = aggregate_timeframe(m1_data, "M3")
        assert len(result) < len(m1_data)
        assert len(result) >= len(m1_data) // 4

    def test_m5_reduces_rows(self, m1_data):
        result = aggregate_timeframe(m1_data, "M5")
        assert len(result) < len(m1_data)

    def test_m7(self, m1_data):
        result = aggregate_timeframe(m1_data, "M7")
        assert len(result) < len(m1_data)
        assert "open" in result.columns

    def test_m14(self, m1_data):
        result = aggregate_timeframe(m1_data, "M14")
        assert len(result) > 0
        assert result["volume"].iloc[0] > 0

    def test_m30(self, m1_data):
        result = aggregate_timeframe(m1_data, "M30")
        assert len(result) > 0

    def test_h1_reduces_rows(self, m1_data):
        result = aggregate_timeframe(m1_data, "H1")
        assert len(result) < len(m1_data)
        assert len(result) >= 1

    def test_h4(self, m1_data):
        result = aggregate_timeframe(m1_data, "H4")
        assert len(result) >= 1

    def test_d1_single_bar_per_day(self, m1_data):
        result = aggregate_timeframe(m1_data, "D1")
        assert len(result) == 1

    def test_d1_high_is_max(self, m1_data):
        result = aggregate_timeframe(m1_data, "D1")
        assert result["high"].iloc[0] >= result["open"].iloc[0]
        assert result["high"].iloc[0] >= result["close"].iloc[0]

    def test_d1_volume_is_sum(self, m1_data):
        result = aggregate_timeframe(m1_data, "D1")
        assert result["volume"].iloc[0] == m1_data["volume"].sum()

    def test_tick_raises(self, m1_data):
        with pytest.raises(ValueError, match="tick"):
            aggregate_timeframe(m1_data, "tick")

    def test_ohlcv_columns_preserved(self, m1_data):
        for tf in ("M3", "M5", "M14", "M30", "H1", "D1"):
            result = aggregate_timeframe(m1_data, tf)
            assert list(result.columns) == ["open", "high", "low", "close", "volume"]


# ======================================================================
# Tick -> bars aggregation
# ======================================================================

class TestTicksToBars:
    @pytest.fixture
    def tick_data(self):
        return load_ticks(FIXTURES_DIR, "LKOH", "2020-03-16", "2020-03-16")

    def test_m1_from_ticks(self, tick_data):
        bars = ticks_to_bars(tick_data, "M1")
        assert len(bars) > 0
        assert list(bars.columns) == ["open", "high", "low", "close", "volume"]

    def test_m5_from_ticks(self, tick_data):
        bars = ticks_to_bars(tick_data, "M5")
        assert len(bars) > 0
        m1_bars = ticks_to_bars(tick_data, "M1")
        assert len(bars) <= len(m1_bars)

    def test_m3_from_ticks(self, tick_data):
        bars = ticks_to_bars(tick_data, "M3")
        assert len(bars) > 0

    def test_h1_from_ticks(self, tick_data):
        bars = ticks_to_bars(tick_data, "H1")
        assert len(bars) >= 1

    def test_d1_from_ticks(self, tick_data):
        bars = ticks_to_bars(tick_data, "D1")
        assert len(bars) == 1

    def test_volume_is_sum(self, tick_data):
        bars = ticks_to_bars(tick_data, "D1")
        assert bars["volume"].iloc[0] == tick_data["volume"].sum()

    def test_float32_dtypes(self, tick_data):
        bars = ticks_to_bars(tick_data, "M5")
        for col in ["open", "high", "low", "close"]:
            assert bars[col].dtype == np.float32

    def test_tick_timeframe_raises(self, tick_data):
        with pytest.raises(ValueError, match="tick"):
            ticks_to_bars(tick_data, "tick")


# ======================================================================
# Unified load_bars
# ======================================================================

class TestLoadBars:
    def test_m1_from_m1(self):
        bars = load_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16",
                         timeframe="M1", raw_source="m1")
        assert len(bars) == 400

    def test_m5_from_m1(self):
        bars = load_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16",
                         timeframe="M5", raw_source="m1")
        assert len(bars) < 400

    def test_d1_from_m1(self):
        bars = load_bars(FIXTURES_DIR, "SBRF", "2020-03-16", "2020-03-16",
                         timeframe="D1", raw_source="m1")
        assert len(bars) == 1

    def test_m5_from_ticks(self):
        bars = load_bars(FIXTURES_DIR, "LKOH", "2020-03-16", "2020-03-16",
                         timeframe="M5", raw_source="ticks")
        assert len(bars) > 0
        assert list(bars.columns) == ["open", "high", "low", "close", "volume"]

    def test_h1_from_ticks(self):
        bars = load_bars(FIXTURES_DIR, "LKOH", "2020-03-16", "2020-03-16",
                         timeframe="H1", raw_source="ticks")
        assert len(bars) >= 1

    def test_tick_as_pseudo_ohlcv(self):
        bars = load_bars(FIXTURES_DIR, "LKOH", "2020-03-16", "2020-03-16",
                         timeframe="tick", raw_source="ticks")
        assert len(bars) == 800
        assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


# ======================================================================
# Scanner
# ======================================================================

class TestScanAvailableTickers:
    def test_finds_tickers(self):
        result = scan_available_tickers(FIXTURES_DIR)
        assert "SBRF" in result
        assert "GMKN" in result
        assert "LKOH" in result

    def test_date_range(self):
        result = scan_available_tickers(FIXTURES_DIR)
        assert result["SBRF"]["days"] == 3
        assert result["GMKN"]["days"] == 2
        assert result["LKOH"]["days"] == 3

    def test_has_m1_flag(self):
        result = scan_available_tickers(FIXTURES_DIR)
        assert result["SBRF"]["has_m1"] is True
        assert result["LKOH"]["has_m1"] is True

    def test_has_ticks_flag(self):
        result = scan_available_tickers(FIXTURES_DIR)
        assert result["SBRF"]["has_ticks"] is True
        assert result["LKOH"]["has_ticks"] is True

    def test_filter_by_date(self):
        result = scan_available_tickers(
            FIXTURES_DIR, start="2020-03-17", end="2020-03-18"
        )
        assert result["SBRF"]["days"] == 2

    def test_empty_dir(self, tmp_path):
        result = scan_available_tickers(str(tmp_path))
        assert len(result) == 0


# ======================================================================
# build_local_panel
# ======================================================================

def _make_panel_cfg(**overrides):
    """Helper: create a Config for panel tests with small indicator periods."""
    defaults = dict(
        ticker="SBRF", start="2020-03-16", end="2020-03-18",
        data_source="local", data_path=FIXTURES_DIR,
        timeframe="M1", local_correlated=("GMKN",),
        use_arima=False, use_sentiment=False,
    )
    defaults.update(overrides)
    return Config(**defaults)


class TestBuildLocalPanel:
    def test_returns_dataframe(self):
        panel = build_local_panel(_make_panel_cfg())
        assert isinstance(panel, pd.DataFrame)
        assert len(panel) > 0

    def test_has_ticker_column(self):
        panel = build_local_panel(_make_panel_cfg())
        assert "SBRF" in panel.columns

    def test_has_correlated_ticker(self):
        panel = build_local_panel(_make_panel_cfg())
        assert "GMKN" in panel.columns

    def test_has_indicators(self):
        panel = build_local_panel(_make_panel_cfg())
        expected = ["sma7", "sma21", "ema21", "macd", "macd_signal",
                    "macd_hist", "bb_upper", "bb_mid", "bb_lower",
                    "rsi14", "mom10", "log_mom10"]
        for col in expected:
            assert col in panel.columns, f"Missing indicator: {col}"

    def test_no_nans(self):
        panel = build_local_panel(_make_panel_cfg())
        assert panel.isna().sum().sum() == 0

    def test_volume_features_when_enabled(self):
        panel = build_local_panel(_make_panel_cfg(use_volume_features=True))
        assert "SBRF_volume" in panel.columns
        assert "SBRF_range" in panel.columns

    def test_invalid_data_path_raises(self):
        cfg = Config(ticker="SBRF", data_source="local",
                     data_path="/nonexistent/path")
        with pytest.raises(RuntimeError, match="not a valid directory"):
            build_local_panel(cfg)

    def test_float32_memory_efficiency(self):
        panel = build_local_panel(_make_panel_cfg())
        float_cols = panel.select_dtypes(include=["float64"]).columns
        assert len(float_cols) == 0, f"Float64 columns found: {list(float_cols)}"

    def test_m5_timeframe(self):
        """Panel built at M5 should have fewer rows than M1."""
        panel_m1 = build_local_panel(_make_panel_cfg(timeframe="M1"))
        panel_m5 = build_local_panel(_make_panel_cfg(timeframe="M5"))
        assert len(panel_m5) < len(panel_m1)

    def test_m14_timeframe(self):
        panel = build_local_panel(_make_panel_cfg(timeframe="M14"))
        assert len(panel) > 0
        assert "sma7" in panel.columns

    def test_from_ticks(self):
        """Panel built from tick data (M1 bars aggregated from ticks)."""
        cfg = _make_panel_cfg(
            ticker="LKOH", local_raw_source="ticks",
            timeframe="M1", local_correlated=(),
        )
        panel = build_local_panel(cfg)
        assert isinstance(panel, pd.DataFrame)
        assert len(panel) > 0
        assert "LKOH" in panel.columns

    def test_data_txt_loaded(self):
        """LKOH uses data.txt files -- verify full panel builds."""
        cfg = _make_panel_cfg(
            ticker="LKOH", local_correlated=("SBRF",),
        )
        panel = build_local_panel(cfg)
        assert len(panel) > 0


# ======================================================================
# Config integration
# ======================================================================

class TestConfigTimeframe:
    def test_timeframe_mult_d1(self):
        assert Config(timeframe="D1").timeframe_mult == 1

    def test_timeframe_mult_h1(self):
        assert Config(timeframe="H1").timeframe_mult == 6  # 390//60

    def test_timeframe_mult_m1(self):
        assert Config(timeframe="M1").timeframe_mult == 390

    def test_timeframe_mult_m5(self):
        assert Config(timeframe="M5").timeframe_mult == 78  # 390//5

    def test_timeframe_mult_m3(self):
        assert Config(timeframe="M3").timeframe_mult == 130  # 390//3

    def test_timeframe_mult_m14(self):
        assert Config(timeframe="M14").timeframe_mult == 27  # 390//14

    def test_apply_defaults_disables_arima_for_m1(self):
        cfg = Config(timeframe="M1", data_source="local")
        cfg.apply_timeframe_defaults()
        assert cfg.use_arima is False

    def test_apply_defaults_disables_arima_for_m5(self):
        cfg = Config(timeframe="M5", data_source="local")
        cfg.apply_timeframe_defaults()
        assert cfg.use_arima is False

    def test_apply_defaults_keeps_arima_for_h1(self):
        cfg = Config(timeframe="H1", data_source="local")
        cfg.apply_timeframe_defaults()
        assert cfg.use_arima is True

    def test_apply_defaults_disables_sentiment_for_local(self):
        cfg = Config(data_source="local")
        cfg.apply_timeframe_defaults()
        assert cfg.use_sentiment is False

    def test_apply_defaults_enables_volume_features(self):
        cfg = Config(data_source="local")
        cfg.apply_timeframe_defaults()
        assert cfg.use_volume_features is True

    def test_wf_params_scale_with_timeframe(self):
        cfg_d1 = Config(timeframe="D1", data_source="local")
        cfg_d1.apply_timeframe_defaults()
        cfg_m5 = Config(timeframe="M5", data_source="local")
        cfg_m5.apply_timeframe_defaults()
        assert cfg_m5.wf_step > cfg_d1.wf_step
        assert cfg_m5.wf_min_train > cfg_d1.wf_min_train
