"""盘前 NaN 尾行必须在数据层被剔除。

真实事故（2026-09-11 ~ 09-24）：云端任务跑在 06:28 ET，盘前 yfinance 会
多给一根全 NaN 的当日行。每个技术信号都读 .iloc[-1]，NaN 参与比较恒为
False，t_trend 于是无条件落到 else 分支判"弱空"。NVDA 技术分 +55 → 0，
综合分少 22 分，而买入门槛是 30——该买的票被静默挡在门外，不报错不告警。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst_v2 import t_macd, t_rsi, t_trend, t_volume


def _uptrend(n: int = 120) -> pd.DataFrame:
    """稳定上升：价 > MA20 > MA50，t_trend 应判多头排列 +2。"""
    close = np.linspace(100.0, 200.0, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


def _with_nan_bar(df: pd.DataFrame) -> pd.DataFrame:
    """复现盘前形态：末尾追加一根全 NaN 的当日行。"""
    nan_row = pd.DataFrame(
        [[np.nan] * len(df.columns)], columns=df.columns,
        index=[df.index[-1] + pd.Timedelta(days=1)],
    )
    return pd.concat([df, nan_row])


class TestNanBarCorruptsSignals:
    """先钉住 bug 本身：没有清洗时，NaN 尾行确实会把多头翻成空头。"""

    def test_trend_flips_bullish_to_bearish_on_nan_bar(self):
        clean = _uptrend()
        assert t_trend(clean)[0] == +2
        assert t_trend(_with_nan_bar(clean))[0] == -1


class TestFetchAllDropsNanBar:
    def test_nan_bar_is_dropped(self, monkeypatch):
        import analyst_v2

        dirty = _with_nan_bar(_uptrend())
        raw = dirty.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"})
        monkeypatch.setattr(analyst_v2.yf, "download", lambda *a, **k: raw)
        monkeypatch.setattr(analyst_v2.yf, "Ticker",
                            lambda s: type("T", (), {"info": {}})())

        df, _ = analyst_v2.fetch_all("TEST")
        assert not df["close"].isna().any()
        # 清洗后信号恢复成数据真实表达的多头
        assert t_trend(df)[0] == +2

    def test_all_nan_raises_rather_than_scoring_garbage(self, monkeypatch):
        import analyst_v2

        n = 5
        raw = pd.DataFrame(
            {c: [np.nan] * n for c in
             ["Open", "High", "Low", "Close", "Volume"]},
            index=pd.date_range("2026-01-01", periods=n, freq="D"))
        monkeypatch.setattr(analyst_v2.yf, "download", lambda *a, **k: raw)
        monkeypatch.setattr(analyst_v2.yf, "Ticker",
                            lambda s: type("T", (), {"info": {}})())

        with pytest.raises(ValueError, match="无有效价格数据"):
            analyst_v2.fetch_all("TEST")


class TestOtherSignalsSurviveCleanData:
    @pytest.mark.parametrize("fn", [t_trend, t_rsi, t_macd, t_volume])
    def test_signal_returns_finite_score(self, fn):
        score, _ = fn(_uptrend())
        assert score == score and -2 <= score <= 2
