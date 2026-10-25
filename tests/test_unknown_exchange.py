"""资产表里出现 SDK 不认识的交易所时，选股不能崩溃。

真实事故（2026-10-20 ~ 10-23）：Alpaca 资产表新增交易所 TXSE，alpaca-py 的
AssetExchange 枚举没有这个值，get_all_assets() 整表校验失败抛异常，
每日任务连续 4 天在选股阶段崩溃，一笔决策都没做。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trading_agent


def _asset(symbol, exchange="NASDAQ", name="Some Corp", tradable=True, fractionable=True):
    return {"symbol": symbol, "exchange": exchange, "name": name,
            "tradable": tradable, "fractionable": fractionable,
            "class": "us_equity", "status": "active"}


class _FakeClient:
    def get(self, path, params=None):
        assert path == "/assets"
        return [
            _asset("AAA"),
            _asset("TXS", exchange="TXSE"),           # SDK 不认识的交易所
            _asset("SPYX", name="Big Index ETF"),      # ETF 应被排除
            _asset("NOFR", fractionable=False),       # 不支持碎股应被排除
        ]


def test_unknown_exchange_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(trading_agent, "UNIVERSE_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "x")

    seen = {}

    class _FakeData:
        def __init__(self, *a, **k):
            pass

        def get_stock_snapshot(self, req):
            seen["symbols"] = list(req.symbol_or_symbols)
            return {}

    monkeypatch.setattr(trading_agent, "StockHistoricalDataClient", _FakeData)

    result = trading_agent.build_universe(_FakeClient())

    assert seen["symbols"] == ["AAA", "TXS"]
    # 快照为空 → 回退到固定观察列表，但不崩溃
    assert result == trading_agent.FALLBACK_WATCHLIST
