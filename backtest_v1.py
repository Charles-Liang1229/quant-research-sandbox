"""
阶段2：回测验证 — 技术面分析师的信号到底有没有预测力？
─────────────────────────────────────────────────────
方法（机构验证因子的标准思路，叫"分位数检验"）：

  1. 取20只不同行业的大盘股，5年历史数据
  2. "穿越"回过去的每一周：只用当天及以前的数据，
     给每只股票算技术面分数（和analyst_v2完全相同的逻辑）
  3. 记录之后20个交易日（约1个月）的真实涨跌
  4. 把所有"分数→未来收益"配对按分数分组：
     如果高分组的未来收益显著高于低分组 → 信号有预测力
     如果各组差不多 → 信号没有预测力（= 我们的系统只是好看的摆设）

两个防作弊要点：
  · 打分只用"当天及以前"的数据（rolling天然满足，无未来函数）
  · 每5个交易日采样一次，降低重叠样本的自相关（避免虚假显著）

用法：python3 backtest_v1.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────
# 参数（都可以改，改完观察结论变不变——稳健性检验）
# ─────────────────────────────────────────────

UNIVERSE = [  # 20只跨行业大盘股（注意：这里有幸存者偏差，后面会讲）
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",   # 科技
    "META", "TSLA", "AMD", "INTC", "NFLX",     # 科技/消费
    "JPM", "V", "GS",                          # 金融
    "JNJ", "PG", "KO", "WMT",                  # 防御/消费
    "XOM", "BA", "DIS",                        # 能源/工业/传媒
]
YEARS = 5            # 回测年数
FWD_DAYS = 20        # 预测窗口：未来20个交易日
SAMPLE_EVERY = 5     # 每5个交易日采样一次（约每周）


# ─────────────────────────────────────────────
# 技术面打分：与 analyst_v2.py 完全相同的逻辑，
# 但改成向量化计算（一次算出每一天的分数，快几百倍）
# ─────────────────────────────────────────────

def tech_score_series(df: pd.DataFrame) -> pd.Series:
    """对整段历史，每天算一个技术面分数（-100 ~ +100）"""
    close, vol = df["close"], df["volume"]

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # 信号1：趋势（权重2.0）
    s_trend = pd.Series(0, index=df.index, dtype=float)
    s_trend[(close > ma20) & (ma20 > ma50)] = +2
    s_trend[(close < ma20) & (ma20 < ma50)] = -2
    mid_up = (close > ma20) & ~((close > ma20) & (ma20 > ma50))
    mid_dn = (close <= ma20) & ~((close < ma20) & (ma20 < ma50))
    s_trend[mid_up] = +1
    s_trend[mid_dn] = -1

    # 信号2：RSI（权重1.0）
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    s_rsi = pd.Series(0, index=df.index, dtype=float)
    s_rsi[rsi > 70] = -1
    s_rsi[(rsi > 55) & (rsi <= 70)] = +1
    s_rsi[(rsi >= 30) & (rsi < 45)] = -1
    s_rsi[rsi < 30] = +1

    # 信号3：MACD（权重1.0）
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    s_macd = pd.Series(-1, index=df.index, dtype=float)
    s_macd[dif > dea] = +1

    # 信号4：量价（权重1.5）
    vratio = vol.rolling(5).mean() / vol.rolling(20).mean()
    ret5 = close.pct_change(5)
    s_vol = pd.Series(0, index=df.index, dtype=float)
    s_vol[(vratio > 1.2) & (ret5 > 0)] = +1
    s_vol[(vratio > 1.2) & (ret5 < 0)] = -1

    # 加权汇总，归一化到 -100 ~ +100（与analyst_v2一致）
    total = s_trend * 2.0 + s_rsi * 1.0 + s_macd * 1.0 + s_vol * 1.5
    max_s = 2 * (2.0 + 1.0 + 1.0 + 1.5)
    return total / max_s * 100


# ─────────────────────────────────────────────
# 回测主流程
# ─────────────────────────────────────────────

def run_backtest():
    print("=" * 66)
    print(f"  技术面信号回测：{len(UNIVERSE)}只股票 × {YEARS}年，预测未来{FWD_DAYS}日")
    print("=" * 66)

    print("\n📥 下载数据中（20只股票，约1分钟）...")
    raw = yf.download(UNIVERSE, period=f"{YEARS}y", progress=False,
                      auto_adjust=True, group_by="ticker")

    records = []
    for sym in UNIVERSE:
        try:
            df = raw[sym][["Close", "Volume"]].dropna().copy()
            df.columns = ["close", "volume"]
        except KeyError:
            print(f"   ⚠️ {sym} 数据缺失，跳过")
            continue
        if len(df) < 300:
            continue

        score = tech_score_series(df)                      # 当天分数（只用过去数据）
        fwd = df["close"].shift(-FWD_DAYS) / df["close"] - 1   # 未来20日收益（这是"答案"）

        # 每5天采样，去掉前60天（均线预热期）和最后20天（没有"未来"可对答案）
        valid = df.index[60:-FWD_DAYS:SAMPLE_EVERY]
        for d in valid:
            records.append((sym, d, score.loc[d], fwd.loc[d]))

    data = pd.DataFrame(records, columns=["sym", "date", "score", "fwd_ret"]).dropna()
    n = len(data)
    print(f"   得到 {n:,} 个『打分 → 未来收益』样本\n")

    # ── 分组检验：按分数分5档，看各档的未来平均收益 ──
    bins   = [-101, -50, -20, 20, 50, 101]
    labels = ["强空(≤-50)", "偏空(-50~-20)", "中性(-20~20)", "偏多(20~50)", "强多(≥50)"]
    data["bucket"] = pd.cut(data["score"], bins=bins, labels=labels)

    print(f"{'信号分组':<16}{'样本数':>8}{'平均未来20日收益':>18}{'胜率(涨的比例)':>16}")
    print("-" * 62)
    stats = {}
    for b in labels:
        grp = data[data["bucket"] == b]["fwd_ret"]
        if len(grp) == 0:
            continue
        stats[b] = grp
        print(f"{b:<16}{len(grp):>8,}{grp.mean():>17.2%}{(grp > 0).mean():>15.1%}")

    bench = data["fwd_ret"].mean()
    print("-" * 62)
    print(f"{'全体平均(基准)':<16}{n:>8,}{bench:>17.2%}{(data['fwd_ret'] > 0).mean():>15.1%}")

    # ── 顶底差 + 简易t检验：差异是真实的还是运气？ ──
    top, bot = stats.get(labels[-1]), stats.get(labels[0])
    print()
    if top is not None and bot is not None and len(top) > 30 and len(bot) > 30:
        spread = top.mean() - bot.mean()
        se = np.sqrt(top.var()/len(top) + bot.var()/len(bot))
        t = spread / se
        print(f"  强多组 − 强空组 收益差：{spread:+.2%}   t统计量 ≈ {t:.2f}")
        print(f"  （经验法则：|t| > 2 才能说『大概率不是运气』）")
        verdict = ("→ 信号有统计上可辨的预测力" if abs(t) > 2
                   else "→ 差异在噪音范围内，不能证明信号有预测力")
        print(f"  {verdict}")

    # ── 信息系数IC：分数和未来收益的秩相关（机构最常用的因子体检指标） ──
    ic = data["score"].corr(data["fwd_ret"], method="spearman")
    print(f"\n  信息系数 IC(Spearman) = {ic:+.3f}")
    print(f"  （机构经验：|IC|>0.03 算有点用，>0.05 算不错，>0.10 算很强）")

    print("\n" + "=" * 66)
    print("  ⚠️ 已知局限（读懂这些和读懂结果一样重要）：")
    print("  1. 幸存者偏差：这20只是『今天还活着的巨头』，天然偏乐观")
    print("  2. 只测了技术面：基本面无历史快照，无法诚实回测")
    print("  3. 未计手续费/滑点：这是信号检验，不是策略收益")
    print("  4. 单一市场单一时段：换个时代（如2008）结论可能不同")
    print("=" * 66)


if __name__ == "__main__":
    run_backtest()
