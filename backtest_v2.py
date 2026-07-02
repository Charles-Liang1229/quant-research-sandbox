"""
阶段2b：信号解剖 — 每个信号 × 每个时间窗口的IC矩阵
─────────────────────────────────────────────────────
上一轮发现：综合分IC=-0.028，没有预测力。
但综合分是四个信号的加权和——问题可能出在"和"上：
    · 也许有的信号是正贡献，被负贡献的抵消了
    · 也许有的信号在5天有效、在60天失效（或反过来）

本轮做法（机构叫"因子体检"）：
    对每个信号单独算IC，在三个预测窗口上：
        5天(短线) / 20天(月度) / 60天(季度)
    得到一张 4信号 × 3窗口 的IC矩阵。

读法提醒：
    IC > 0：信号方向和未来收益一致（"多头信号→真的涨"）
    IC < 0：信号方向和未来收益相反（信号反着用反而对？小心！）
    |IC| < 0.03：噪音，当它不存在

用法：python3 backtest_v2.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "INTC", "NFLX",
    "JPM", "V", "GS",
    "JNJ", "PG", "KO", "WMT",
    "XOM", "BA", "DIS",
]
YEARS = 5
HORIZONS = [5, 20, 60]       # 三个预测窗口（交易日）
SAMPLE_EVERY = 5


# ─────────────────────────────────────────────
# 四个信号，各自独立输出分数序列
# （逻辑与 analyst_v2 一致，但这次不加权汇总）
# ─────────────────────────────────────────────

def sig_trend(df):
    """均线排列：+2/-2多空排列，+1/-1弱多弱空"""
    close = df["close"]
    ma20, ma50 = close.rolling(20).mean(), close.rolling(50).mean()
    s = pd.Series(0.0, index=df.index)
    s[close > ma20] = +1
    s[close <= ma20] = -1
    s[(close > ma20) & (ma20 > ma50)] = +2
    s[(close < ma20) & (ma20 < ma50)] = -2
    return s


def sig_rsi(df):
    """RSI区间打分：超买-1 / 偏强+1 / 偏弱-1 / 超卖+1"""
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    s = pd.Series(0.0, index=df.index)
    s[rsi > 70] = -1          # 超买 → 看空（均值回归假设）
    s[(rsi > 55) & (rsi <= 70)] = +1   # 偏强 → 看多（动量假设）
    s[(rsi >= 30) & (rsi < 45)] = -1   # 偏弱 → 看空
    s[rsi < 30] = +1          # 超卖 → 看多（均值回归假设）
    return s


def sig_macd(df):
    """MACD：金叉+1 / 死叉-1"""
    close = df["close"]
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    s = pd.Series(-1.0, index=df.index)
    s[dif > dea] = +1
    return s


def sig_volume(df):
    """量价：放量涨+1 / 放量跌-1 / 其他0"""
    vratio = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()
    ret5 = df["close"].pct_change(5)
    s = pd.Series(0.0, index=df.index)
    s[(vratio > 1.2) & (ret5 > 0)] = +1
    s[(vratio > 1.2) & (ret5 < 0)] = -1
    return s


SIGNALS = {
    "趋势(均线)": sig_trend,
    "RSI":        sig_rsi,
    "MACD":       sig_macd,
    "量价":       sig_volume,
}


# ─────────────────────────────────────────────
# 主流程：构建 信号×窗口 的样本，算IC矩阵
# ─────────────────────────────────────────────

def main():
    print("=" * 66)
    print(f"  信号解剖：{len(SIGNALS)}个信号 × {HORIZONS}日窗口 × {len(UNIVERSE)}只股票 × {YEARS}年")
    print("=" * 66)
    print("\n📥 下载数据中 ...")

    raw = yf.download(UNIVERSE, period=f"{YEARS}y", progress=False,
                      auto_adjust=True, group_by="ticker")

    # rows[信号名][窗口] = list of (分数, 未来收益)
    rows = {name: {h: [] for h in HORIZONS} for name in SIGNALS}

    for sym in UNIVERSE:
        try:
            df = raw[sym][["Close", "Volume"]].dropna().copy()
            df.columns = ["close", "volume"]
        except KeyError:
            continue
        if len(df) < 300:
            continue

        scores = {name: fn(df) for name, fn in SIGNALS.items()}
        fwd = {h: df["close"].shift(-h) / df["close"] - 1 for h in HORIZONS}

        max_h = max(HORIZONS)
        valid = df.index[60:-max_h:SAMPLE_EVERY]
        for name in SIGNALS:
            for h in HORIZONS:
                s, f = scores[name].loc[valid], fwd[h].loc[valid]
                mask = s.notna() & f.notna()
                rows[name][h].extend(zip(s[mask], f[mask]))

    n_samples = len(rows[list(SIGNALS)[0]][HORIZONS[0]])
    print(f"   每个格子约 {n_samples:,} 个样本\n")

    # ── IC矩阵 ──
    header = f"{'信号':<12}" + "".join(f"{f'{h}日IC':>12}" for h in HORIZONS)
    print(header)
    print("-" * len(header.expandtabs()))

    def fmt(ic, n):
        """IC + 显著性标记。secure rough t ≈ IC*sqrt(N)，|t|>2 标*"""
        t = ic * np.sqrt(n)
        mark = " *" if abs(t) > 2 else "  "
        return f"{ic:+.3f}{mark}"

    for name in SIGNALS:
        cells = []
        for h in HORIZONS:
            pairs = rows[name][h]
            s = pd.Series([p[0] for p in pairs])
            f = pd.Series([p[1] for p in pairs])
            ic = s.corr(f, method="spearman")
            cells.append(fmt(ic, len(pairs)))
        print(f"{name:<12}" + "".join(f"{c:>12}" for c in cells))

    print("\n  读法：* = 粗略t检验显著(|t|>2)；|IC|<0.03 ≈ 噪音")
    print("  IC为负 = 信号和未来收益方向相反（例如『金叉后反而跌』）")

    print("\n" + "=" * 66)
    print("  ⚠️ 同样的局限依然存在：幸存者偏差 / 无费用 / 单一时段。")
    print("  另外注意：我们一口气测了12个格子——测的越多，越容易碰上")
    print("  『纯靠运气显著』的格子（多重检验问题）。对任何单个 * 都别太激动，")
    print("  真正的验证是：换一批股票、换一个时段，它还在不在。")
    print("=" * 66)


if __name__ == "__main__":
    main()
