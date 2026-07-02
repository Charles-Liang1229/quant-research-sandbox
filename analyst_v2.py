"""
阶段1b：双分析师系统（技术面 + 基本面）v2
─────────────────────────────────────────────────────
架构升级：
    v1 = 5个技术信号 → 一个记分卡
    v2 = 技术面分析师 + 基本面分析师 → 各自记分卡 → 总经理汇总

每个信号依然带三段说明：
    做什么(What) / 为什么有效(Why) / 什么时候失效(When it fails)

用法：python3 analyst_v2.py INTC
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import ta


# ═════════════════════════════════════════════
# 数据层：一次性取回技术面 + 基本面数据
# ═════════════════════════════════════════════

def fetch_all(symbol: str):
    tk = yf.Ticker(symbol)
    raw = yf.download(symbol, period="310d", progress=False, auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    price_df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    price_df.columns = ["open", "high", "low", "close", "volume"]
    info = tk.info   # 基本面快照（PE、利润率、增长等）
    return price_df.astype(float), info


# ═════════════════════════════════════════════
# 分析师A：技术面（沿用v1的核心信号）
# ═════════════════════════════════════════════

def t_trend(df):
    """趋势-均线排列。失效场景：横盘震荡市反复假信号。"""
    price = df["close"].iloc[-1]
    ma20 = df["close"].rolling(20).mean().iloc[-1]
    ma50 = df["close"].rolling(50).mean().iloc[-1]
    if price > ma20 > ma50:
        return +2, f"多头排列 价{price:.1f}>MA20 {ma20:.1f}>MA50 {ma50:.1f}"
    if price < ma20 < ma50:
        return -2, f"空头排列 价{price:.1f}<MA20 {ma20:.1f}<MA50 {ma50:.1f}"
    return (+1, "价在MA20上方，弱多") if price > ma20 else (-1, "价在MA20下方，弱空")


def t_rsi(df):
    """动量-RSI。失效场景：强趋势中长期钝化。"""
    rsi = ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1]
    if rsi > 70: return -1, f"RSI={rsi:.0f} 超买（趋势强时可能钝化）"
    if rsi < 30: return +1, f"RSI={rsi:.0f} 超卖"
    if rsi > 55: return +1, f"RSI={rsi:.0f} 偏强"
    if rsi < 45: return -1, f"RSI={rsi:.0f} 偏弱"
    return 0, f"RSI={rsi:.0f} 中性"


def t_macd(df):
    """MACD动能。失效场景：横盘时金叉死叉全是噪音。"""
    m = ta.trend.MACD(df["close"])
    dif, dea = m.macd().iloc[-1], m.macd_signal().iloc[-1]
    if dif > dea: return +1, f"金叉 DIF{dif:.2f}>DEA{dea:.2f}"
    return -1, f"死叉 DIF{dif:.2f}<DEA{dea:.2f}"


def t_volume(df):
    """量价配合。失效场景：财报等一次性事件的异常放量。"""
    ratio = df["volume"].rolling(5).mean().iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
    ret5 = df["close"].pct_change(5).iloc[-1]
    if ratio > 1.2 and ret5 > 0: return +1, f"放量上涨({ratio:.1f}x)"
    if ratio > 1.2 and ret5 < 0: return -1, f"放量下跌({ratio:.1f}x)"
    return 0, f"量能正常({ratio:.1f}x)"


# ═════════════════════════════════════════════
# 分析师B：基本面（新增）
# ═════════════════════════════════════════════

def f_valuation(info):
    """
    【估值 — 市盈率PE】
    做什么：股价 ÷ 每股盈利。你为公司每$1利润付多少钱。
    为什么有效：长期看股价围绕价值波动。PE过高=透支未来，
        PE过低=可能被低估（也可能有雷）。
    什么时候失效：
        ① 亏损公司PE为负/无意义（转型期的Intel就常这样）；
        ② 周期股在利润顶点PE最低——恰恰是最危险的时候（"低PE陷阱"）；
        ③ 高成长公司PE永远"贵"，光看PE会错过所有大牛股。
    """
    pe = info.get("trailingPE")
    fpe = info.get("forwardPE")
    if pe is None or pe <= 0:
        if fpe and fpe > 0:
            return 0, f"当前无有效PE（可能亏损），远期PE={fpe:.0f}（市场在赌未来扭亏）"
        return 0, "PE不可用（亏损或数据缺失）——估值锚失效，风险更高"
    note = f"PE={pe:.0f}"
    if fpe and fpe > 0:
        note += f"，远期PE={fpe:.0f}"
        if fpe < pe * 0.7:
            return +1, note + " → 市场预期利润明显增长"
    if pe < 15: return +1, note + " → 传统意义偏便宜（警惕低PE陷阱）"
    if pe > 40: return -1, note + " → 偏贵，增长必须兑现否则杀估值"
    return 0, note + " → 估值中性"


def f_profitability(info):
    """
    【盈利质量 — 利润率 & ROE】
    做什么：净利润率(赚钱效率) + 净资产收益率ROE(股东资金回报率)。
    为什么有效：长期股价跟随盈利能力。高利润率=有定价权/护城河；
        巴菲特最看重ROE——持续>15%的公司往往是复利机器。
    什么时候失效：
        ① 高杠杆能人为推高ROE（借钱放大），要配合负债看；
        ② 重资产转型期（如Intel建厂）利润率被压制，反映的是
          投入期而非真实盈利能力——用当期利润率判断会误杀。
    """
    margin = info.get("profitMargins")
    roe = info.get("returnOnEquity")
    parts, score = [], 0
    if margin is not None:
        parts.append(f"净利率={margin*100:.1f}%")
        score += 1 if margin > 0.15 else (-1 if margin < 0.02 else 0)
    if roe is not None:
        parts.append(f"ROE={roe*100:.1f}%")
        score += 1 if roe > 0.15 else (-1 if roe < 0.05 else 0)
    if not parts:
        return 0, "盈利数据缺失"
    score = max(-2, min(2, score))
    return score, "，".join(parts)


def f_growth(info):
    """
    【成长性 — 营收增速】
    做什么：最近季度营收同比增长率。
    为什么有效：营收比利润更难做假账（利润可通过会计手段调节）。
        营收持续增长是公司扩张最直接的证据。
    什么时候失效：
        ① 单季数据受基数效应影响大（去年同期特别差→今年"高增长"）；
        ② 靠降价换来的营收增长会毁掉利润率——要配合利润率一起看；
        ③ 并购带来的增长不是内生增长。
    """
    g = info.get("revenueGrowth")
    if g is None:
        return 0, "营收增速数据缺失"
    note = f"营收同比{g*100:+.1f}%"
    if g > 0.15: return +2, note + " → 高增长"
    if g > 0.05: return +1, note + " → 稳健增长"
    if g < -0.05: return -2, note + " → 营收萎缩，警惕"
    return 0, note + " → 基本持平"


def f_balance(info):
    """
    【财务安全 — 负债与现金流】
    做什么：负债/股东权益比(D/E) + 自由现金流(FCF)。
    为什么有效：牛市里没人看负债，熊市里负债决定谁活下来。
        FCF是公司真正能自由支配的钱——分红、回购、还债全靠它。
        利润可以"纸面"，现金流很难造假。
    什么时候失效：
        ① 不同行业负债水平天差地别（银行/公用事业天然高杠杆），
          跨行业直接比D/E会得出错误结论；
        ② 重投资期FCF为负不一定是坏事（Intel建厂烧钱换未来产能）——
          关键是烧的钱未来能不能长出回报。
    """
    de = info.get("debtToEquity")
    fcf = info.get("freeCashflow")
    parts, score = [], 0
    if de is not None:
        parts.append(f"D/E={de:.0f}%")
        score += -1 if de > 150 else (1 if de < 50 else 0)
    if fcf is not None:
        parts.append(f"FCF={fcf/1e9:+.1f}B")
        score += 1 if fcf > 0 else -1
    if not parts:
        return 0, "财务数据缺失"
    score = max(-2, min(2, score))
    return score, "，".join(parts)


# ═════════════════════════════════════════════
# 总经理层：汇总两位分析师
# ═════════════════════════════════════════════

def run_analyst(title, signals, data):
    """跑一位分析师的所有信号，返回归一化分数(-100~+100)"""
    print(f"\n┌─ {title}")
    total, max_s = 0.0, 0.0
    for name, func, weight in signals:
        score, note = func(data)
        total += score * weight
        max_s += 2 * weight
        print(f"│ {name:<12}{f'{score:+d}×{weight}':<8}{note}")
    norm = total / max_s * 100 if max_s else 0
    print(f"└─ 小计：{norm:+.0f}/100")
    return norm


def analyze(symbol: str):
    df, info = fetch_all(symbol)
    price = df["close"].iloc[-1]
    name = info.get("shortName", symbol)

    print("=" * 62)
    print(f"  双分析师系统 v2 — {name} ({symbol})  现价 ${price:.2f}")
    print("=" * 62)

    tech_signals = [
        ("趋势(均线)", t_trend, 2.0),
        ("动量(RSI)",  t_rsi,   1.0),
        ("MACD",       t_macd,  1.0),
        ("量价",       t_volume, 1.5),
    ]
    fund_signals = [
        ("估值(PE)",   f_valuation,     1.5),
        ("盈利质量",   f_profitability, 1.5),
        ("成长性",     f_growth,        1.5),
        ("财务安全",   f_balance,       1.0),
    ]

    tech = run_analyst("技术面分析师（回答：现在时机如何？）", tech_signals, df)
    fund = run_analyst("基本面分析师（回答：公司值不值得拥有？）", fund_signals, info)

    # 总经理汇总：这里的权重也是"观点"——长线投资者应加重基本面
    W_TECH, W_FUND = 0.2, 0.8
    combined = tech * W_TECH + fund * W_FUND

    print("\n" + "=" * 62)
    print(f"  总经理汇总（技术{W_TECH:.0%} + 基本面{W_FUND:.0%}）")
    print("=" * 62)
    print(f"  技术面 {tech:+.0f} × {W_TECH} + 基本面 {fund:+.0f} × {W_FUND} = 综合 {combined:+.0f}/100")
    print(f"\n  {_verdict(tech, fund)}")
    print("\n  ⚠️ 研究工具，非投资建议。数据源为yfinance免费接口，可能有延迟或缺失。")
    print("=" * 62)


def _verdict(tech, fund):
    if tech > 20 and fund > 20:
        return "双面共振偏多：时机与质地都不差 → 值得深入研究（而非直接买入）"
    if tech > 20 and fund <= 0:
        return "技术强但基本面弱 → 典型的『炒作/预期驱动』行情，涨得快也跌得快，短线思维、严格止损"
    if tech <= 0 and fund > 20:
        return "基本面好但技术弱 → 好公司暂时不受待见，长线者的观察名单，等技术面企稳"
    if tech < -20 and fund < -20:
        return "双面共振偏空 → 回避"
    return "信号分歧或均不显著 → 最佳操作往往是：等待，把钱留给更清晰的机会"


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "MU"
    analyze(symbol)
