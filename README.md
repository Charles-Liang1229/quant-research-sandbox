# Quant Research Sandbox

A hands-on learning project: build a multi-signal stock analysis system, then **rigorously test whether it actually works** — and accept the answer the data gives.

> **TL;DR result:** I built a technical + fundamental scoring system, backtested its signals on 5 years of US large-cap data, and found **no exploitable predictive power** (IC ≈ -0.03). The only statistically detectable pattern pointed the *opposite* way — weak mean reversion. The negative result is the point: most signals fail, and knowing how to prove that is the actual skill.

---

## Project structure

| File | Stage | What it does |
|------|-------|--------------|
| `quant_demo.py` | 0 | First end-to-end ML pipeline: XGBoost predicting next-day direction on AAPL, with feature engineering, time-series split, and a fee-aware backtest |
| `analyst_v1.py` | 1 | Transparent rule-based technical analyst: 5 signals (trend, RSI, MACD, volume, ATR), each documented with *what it does / why it works / when it fails* |
| `analyst_v2.py` | 1b | Two-analyst system: technical analyst ("is the timing right?") + fundamental analyst ("is the company worth owning?") → weighted GM summary |
| `backtest_v1.py` | 2 | Quantile test: does the composite technical score predict 20-day forward returns? (20 large caps × 5 years, ~4,700 samples) |
| `backtest_v2.py` | 2b | Signal dissection: per-signal Information Coefficient across 5/20/60-day horizons — a 4×3 factor health-check matrix |

## How to run

```bash
pip3 install yfinance ta pandas numpy scikit-learn xgboost matplotlib

python3 analyst_v2.py NVDA      # score any ticker
python3 backtest_v1.py          # composite-score quantile test
python3 backtest_v2.py          # per-signal IC matrix
```

Requires `libomp` on macOS for XGBoost (`brew install libomp`).

---

## Key findings

### 1. The composite score has no predictive power

Quantile test on ~4,700 (score → 20-day forward return) samples:

| Signal bucket | Avg fwd return | Win rate |
|---------------|---------------:|---------:|
| Strong bear (≤ -50) | +1.82% | 57.7% |
| Bear (-50 ~ -20) | +1.79% | 57.1% |
| Neutral | +3.14% | 59.6% |
| Bull (20 ~ 50) | +1.71% | 54.6% |
| Strong bull (≥ 50) | +1.11% | 53.6% |

Top-minus-bottom spread: **-0.71%** (t ≈ -1.26, not significant). Spearman IC: **-0.028**.

### 2. Dissecting it: every component fails individually

| Signal | 5d IC | 20d IC | 60d IC |
|--------|------:|-------:|-------:|
| Trend (MA alignment) | -0.024 | **-0.036*** | -0.014 |
| RSI | -0.004 | +0.002 | +0.017 |
| MACD | -0.000 | -0.028 | **-0.034*** |
| Volume-price | +0.004 | +0.010 | -0.004 |

(* = rough t-test |t| > 2)

No signal shows positive predictive power. The only significant cells are **negative** — classic trend-following signals mildly *reverse* on US large caps at monthly horizons, consistent with the quantile test. Evidence of weak mean reversion, but at |IC| ≈ 0.03 it would not survive transaction costs.

### 3. Lessons that survived contact with the data

- **Look-ahead bias is everywhere by default.** Labels must be strictly shifted; fundamentals can't be honestly backtested without point-in-time data (today's PE snapshot ≠ what the market saw in 2023).
- **Everything "makes money" in a bull-market backtest.** All five buckets averaged +1.7%/month — that's beta plus survivorship bias, not signal. Always compare against the base rate.
- **Multiple testing inflates false positives.** 12 cells tested → a couple of significant-looking cells are expected by chance. A finding is real only if it survives a new universe and a new period.
- **Free data lies a little.** yfinance prices diverged from my broker's quotes; cross-checking data sources comes before trusting any model built on them.
- **A scoring system without validation is a checklist, not a predictor.** Useful for disciplined analysis and for flagging conflicting evidence — not for sizing positions.

---

## Known limitations

1. **Survivorship bias** — the 20-stock universe is today's mega-cap winners
2. **Technical signals only** — no point-in-time fundamental database
3. **No transaction costs / slippage** in the IC tests (signal quality, not strategy P&L)
4. **Single market, single regime** — 5 recent years of US large caps; conclusions may not transfer

## Possible next steps

- Test the mean-reversion hypothesis out-of-sample (different universe, different period)
- Add a point-in-time fundamental dataset and extend the IC matrix
- Wrap the analysts as independent agents with an orchestration layer (original motivation for this project)

---

*Built as a learning project — the goal was never a trading edge, but the research muscle: hypothesize → test → dissect → let the data overrule you.*

---

## v4：AI 主题股池（与 autotrader 的对照实验）

Agent 的观察列表不再是硬编码的 10 只大盘股。股池按三层降级取用：

1. **AI 主题股池**（首选）：deer-flow 仓库的 autotrader 每周采集新闻/宏观/
   国会政策/行业轮动后由 Claude 生成 `universe.json`（约 30 只，含回避清单）。
   本地运行直接读取；云端 GitHub Actions 使用仓库内快照 `universe_ai.json`。
2. **全市场成交额筛选**（兜底）：AI 股池缺失或超过 21 天未更新时，
   从 Alpaca 全部可交易美股按当日成交额取前 100。
3. **固定列表**（保底）：以上都失败时用原 10 只。

这构成一个对照实验：**同一个股池**，本 agent 用"技术+基本面综合打分"
（回测已证明无预测优势的旧大脑），autotrader 用"趋势+动量"机械规则。
几周模拟盘成绩对比后，用数据决定谁的选股逻辑留下。

其他改动：打分器明确跳过 ETF（基本面字段不适用）；持仓打分失败时按
持有处理而非误卖；每日决策日志记录当天使用的股池来源。
