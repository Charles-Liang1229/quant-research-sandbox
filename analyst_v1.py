"""
阶段1：透明规则型股票分析师（Rule-based Analyst v1）
─────────────────────────────────────────────────────
目标：不用LLM，先做一个你能看懂每一行逻辑的分析器。
每个"信号"都是一个独立函数，配三段说明：
    做什么(What) / 为什么有效(Why) / 什么时候失效(When it fails)

这样你能：看懂 → 修改 → 判断对不对。

用法：python3 analyst_v1.py AAPL
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import ta


# ─────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────

def fetch(symbol: str, period_days: int = 250) -> pd.DataFrame:
    """拉取约1年日线。250 ≈ 一年的交易日数量。"""
    raw = yf.download(symbol, period=f"{period_days+60}d",
                      progress=False, auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df.astype(float)


# ─────────────────────────────────────────────
# 信号函数：每个返回 (信号分数, 一句话解读)
# 分数区间 -2(强空) ~ +2(强多)
# ─────────────────────────────────────────────

def signal_trend(df: pd.DataFrame) -> tuple[int, str]:
    """
    【趋势 — 均线排列】
    做什么：比较收盘价与20日、50日均线的相对位置。
    为什么有效：均线是"过去N天平均成本"。价在均线上方 =
        近期买入者普遍浮盈、抛压小，趋势向上；反之向下。
        这是最古老也最稳健的趋势判断，机构都在用。
    什么时候失效：横盘震荡市里，价格反复穿越均线 → 频繁假信号。
        均线是"滞后指标"，转折点它永远慢半拍。
    """
    price = df["close"].iloc[-1]
    ma20 = df["close"].rolling(20).mean().iloc[-1]
    ma50 = df["close"].rolling(50).mean().iloc[-1]

    if price > ma20 > ma50:
        return +2, f"价({price:.1f}) > MA20({ma20:.1f}) > MA50({ma50:.1f})，多头排列，趋势明确向上"
    if price < ma20 < ma50:
        return -2, f"价({price:.1f}) < MA20({ma20:.1f}) < MA50({ma50:.1f})，空头排列，趋势向下"
    if price > ma20:
        return +1, f"价在MA20上方但均线未完全多头排列，弱势偏多"
    return -1, "价在MA20下方，弱势偏空"


def signal_momentum(df: pd.DataFrame) -> tuple[int, str]:
    """
    【动量 — RSI 相对强弱】
    做什么：计算14日RSI，衡量近期涨跌力量对比（0~100）。
    为什么有效：>70 表示买盘过热（可能超买回调），
        <30 表示卖盘过度（可能超卖反弹）。它捕捉"情绪极端"。
    什么时候失效：强趋势中RSI会长期钝化——大牛股RSI能在70以上
        待好几周还继续涨。这时"超买=该卖"就是错的。
        所以RSI在震荡市好用，在单边市要小心。
    """
    rsi = ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1]
    if rsi > 70:
        return -1, f"RSI={rsi:.0f}，超买区，短期回调风险（但强趋势中可能钝化，别急着看空）"
    if rsi < 30:
        return +1, f"RSI={rsi:.0f}，超卖区，可能反弹"
    if rsi > 55:
        return +1, f"RSI={rsi:.0f}，动量偏强"
    if rsi < 45:
        return -1, f"RSI={rsi:.0f}，动量偏弱"
    return 0, f"RSI={rsi:.0f}，中性"


def signal_macd(df: pd.DataFrame) -> tuple[int, str]:
    """
    【趋势动量 — MACD】
    做什么：用快慢两条指数均线的差(DIF)与其信号线(DEA)判断动能转变。
    为什么有效：DIF上穿DEA(金叉)=短期动能转强；下穿(死叉)=转弱。
        它比单纯均线更早捕捉到"加速/减速"。
    什么时候失效：同样是滞后指标；在无趋势的横盘里，金叉死叉
        会来回出现，全是噪音。
    """
    macd = ta.trend.MACD(df["close"])
    dif = macd.macd().iloc[-1]
    dea = macd.macd_signal().iloc[-1]
    hist = macd.macd_diff().iloc[-1]

    if dif > dea and hist > 0:
        return +1, f"MACD金叉(DIF {dif:.2f} > DEA {dea:.2f})，动能偏多"
    if dif < dea and hist < 0:
        return -1, f"MACD死叉(DIF {dif:.2f} < DEA {dea:.2f})，动能偏空"
    return 0, "MACD纠缠，方向不明"


def signal_volume(df: pd.DataFrame) -> tuple[int, str]:
    """
    【量价配合】
    做什么：比较最近5日均量与过去20日均量。
    为什么有效："量在价先"——真正的趋势需要成交量支持。
        放量上涨=资金真进场；缩量上涨=可能是虚涨、无人接力。
    什么时候失效：财报、突发新闻会造成异常放量，这种量不代表
        趋势，是一次性事件。单看量容易误读。
    """
    vol5 = df["volume"].rolling(5).mean().iloc[-1]
    vol20 = df["volume"].rolling(20).mean().iloc[-1]
    ratio = vol5 / vol20
    ret5 = df["close"].pct_change(5).iloc[-1]

    if ratio > 1.2 and ret5 > 0:
        return +1, f"近5日放量({ratio:.1f}倍)且上涨，资金进场，量价配合良好"
    if ratio > 1.2 and ret5 < 0:
        return -1, f"近5日放量({ratio:.1f}倍)但下跌，抛压重"
    return 0, f"成交量正常({ratio:.1f}倍均量)，无明显异动"


def signal_volatility(df: pd.DataFrame) -> tuple[int, str]:
    """
    【波动率 — ATR】（注意：这不是方向信号，是"风险温度计"）
    做什么：计算14日平均真实波幅占价格的百分比。
    为什么有效：告诉你这只票平时一天能晃多少。ATR高=风险大，
        直接影响你该设多宽的止损、下多大的仓位。
    什么时候失效：它只测"晃得凶不凶"，不测"往哪晃"。别拿它当
        买卖信号，它是用来做"仓位和止损"的辅助工具。
    """
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]
    price = df["close"].iloc[-1]
    atr_pct = atr / price * 100
    return 0, f"ATR={atr:.2f}（日均波动{atr_pct:.1f}%）→ 建议止损宽度≈{2*atr:.1f}点(2倍ATR)"


# ─────────────────────────────────────────────
# 汇总：把各信号加权成一个透明记分卡
# ─────────────────────────────────────────────

def analyze(symbol: str):
    df = fetch(symbol)
    price = df["close"].iloc[-1]

    print("=" * 60)
    print(f"  透明规则型分析师 v1  —  {symbol}  当前价 ${price:.2f}")
    print("=" * 60)

    # 每个信号 (名字, 权重)。权重体现"你有多信它"——这是你能调的地方
    signals = [
        ("趋势(均线)",   signal_trend,      2.0),
        ("动量(RSI)",    signal_momentum,   1.0),
        ("MACD",         signal_macd,       1.0),
        ("量价",         signal_volume,     1.5),
        ("波动率(ATR)",  signal_volatility, 0.0),  # 权重0：不计入方向，仅供参考
    ]

    total_score = 0.0
    max_score = 0.0
    print(f"\n{'信号':<14}{'分数':<6}{'解读'}")
    print("-" * 60)
    for name, func, weight in signals:
        score, note = func(df)
        total_score += score * weight
        max_score += 2 * weight   # 每个信号最高+2
        weighted = f"{score:+d}×{weight}"
        print(f"{name:<14}{weighted:<8}{note}")

    # 归一化到 -100 ~ +100，方便直觉理解
    norm = total_score / max_score * 100 if max_score else 0

    print("-" * 60)
    print(f"\n综合评分：{norm:+.0f} / 100")
    print(_verdict(norm))
    print("\n⚠️  这是研究工具，不是买卖指令。最终决定永远是你自己。")
    print("=" * 60)


def _verdict(norm: float) -> str:
    if norm >= 50:
        return "→ 技术面偏多：多个信号共振向上。但注意别在情绪高点追高。"
    if norm >= 20:
        return "→ 弱势偏多：有一些多头信号，但不够强，观察为主。"
    if norm > -20:
        return "→ 中性/纠缠：信号打架，方向不明，此时最该做的是等待。"
    if norm > -50:
        return "→ 弱势偏空：多个信号转弱，谨慎。"
    return "→ 技术面偏空：信号共振向下，回避或考虑止损。"


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "INTC"
    analyze(symbol)
