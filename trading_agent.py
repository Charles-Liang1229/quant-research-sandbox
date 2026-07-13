"""
阶段3：自主交易Agent（模拟盘 Paper Trading）
─────────────────────────────────────────────────────
架构（每次运行执行一个完整决策循环）：

  ① 调研员  拉取观察列表的行情与基本面
  ② 分析师  用 analyst_v2 的双分析师逻辑给每只股票打分
  ③ 风控官  硬编码的仓位与风险规则（AI无权越过）
  ④ 执行员  通过 Alpaca API 在模拟盘下单
  ⑤ 记录员  每个决策连同理由写入 decisions_log.csv

风控规则（写死在代码里，这是agent的"宪法"）：
  · 单只持仓上限：组合的15%
  · 每次建仓规模：组合的10%
  · 总持仓上限：组合的80%（永远留20%现金）
  · 止损：跌破买入价8% → 强制卖出
  · 每天最多新开2个仓位（防抽风）
  · 只用现金，不碰杠杆（尽管账户给了4倍购买力）

诚实声明：
  这个agent的"大脑"（打分逻辑）经回测证明【没有】预测优势
  （见README的IC分析）。运行它的目的是搭建和验证基础设施，
  以及积累一份真实的模拟盘业绩记录——不是赚钱。

用法：
  python3 trading_agent.py           # 跑一个完整决策循环
  python3 trading_agent.py status    # 只看当前持仓和盈亏
"""

import os
import sys
import csv
import json
import warnings
from datetime import datetime, timezone, date

warnings.filterwarnings("ignore")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest

# 复用双分析师的打分逻辑（同目录的 analyst_v2.py）
from analyst_v2 import (fetch_all, t_trend, t_rsi, t_macd, t_volume,
                        f_valuation, f_profitability, f_growth, f_balance)

# ─────────────────────────────────────────────
# 配置（观察列表和阈值可调；风控规则不建议放松）
# ─────────────────────────────────────────────

# 股池三层降级：AI主题股池（首选，见①a）→ 全市场成交额前N（①b兜底）
# → FALLBACK_WATCHLIST（最后保底），保证每日任务永不中断。
UNIVERSE_SIZE      = 100    # 兜底筛选的股票数（打分一只约1-2秒，100只约3-5分钟）
MIN_PRICE          = 5.0    # 低于$5的仙股不碰（流动性差、数据脏、操纵多）
UNIVERSE_CACHE     = os.path.join(os.path.dirname(__file__), "universe_cache.json")

FALLBACK_WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                      "META", "JPM", "V", "XOM", "WMT"]

BUY_THRESHOLD  = 30     # 综合分 ≥ 30 才考虑买入
SELL_THRESHOLD = 0      # 持仓股综合分 ≤ 0 → 卖出
W_TECH, W_FUND = 0.4, 0.6

# ── 风控宪法（改小可以，改大前先问自己：回测支持吗？） ──
MAX_POSITION_PCT   = 0.15   # 单只上限15%
ENTRY_SIZE_PCT     = 0.10   # 每次建仓10%
MAX_EXPOSURE_PCT   = 0.80   # 总仓位上限80%
STOP_LOSS_PCT      = 0.08   # 止损线8%
MAX_NEW_BUYS_A_DAY = 2      # 每天最多新开2仓

LOG_FILE = os.path.join(os.path.dirname(__file__), "decisions_log.csv")


# ─────────────────────────────────────────────
# ② 分析师：给一只股票打综合分（静默版，不打印过程）
# ─────────────────────────────────────────────

def score_stock(symbol: str) -> tuple[float, str]:
    df, info = fetch_all(symbol)
    # AI股池里可能混入行业ETF（XLV/SMH等）。本agent的基本面打分依赖
    # PE/利润率/增长，ETF没有这些数据，硬算只会得到垃圾分 → 明确跳过。
    quote_type = info.get("quoteType")
    if quote_type not in (None, "EQUITY"):
        raise ValueError(f"非个股({quote_type})，基本面打分不适用")
    tech_signals = [(t_trend, 2.0), (t_rsi, 1.0), (t_macd, 1.0), (t_volume, 1.5)]
    fund_signals = [(f_valuation, 1.5), (f_profitability, 1.5),
                    (f_growth, 1.5), (f_balance, 1.0)]

    def subtotal(signals, data):
        total = sum(fn(data)[0] * w for fn, w in signals)
        max_s = sum(2 * w for _, w in signals)
        return total / max_s * 100

    tech = subtotal(tech_signals, df)
    fund = subtotal(fund_signals, info)
    combined = tech * W_TECH + fund * W_FUND
    price = df["close"].iloc[-1]
    return combined, f"tech={tech:+.0f} fund={fund:+.0f} price=${price:.2f}"


# ─────────────────────────────────────────────
# ①a 调研员：AI 主题股池（首选）
# ─────────────────────────────────────────────
# autotrader（deer-flow 仓库）每周由 Claude 采集新闻/宏观/政策后生成
# universe.json：主题股池 + 回避清单。本 agent 优先使用同一股池——
# 两个模拟盘由此构成对照实验：同一个股池，
#   本 agent   = 技术+基本面综合打分（旧大脑）
#   autotrader = 趋势+动量机械选股（新大脑）
# 几周后对比成绩，用数据决定谁的选股逻辑留下。
#
# 查找顺序（第一个可用的生效）：
#   1. 环境变量 UNIVERSE_FILE 指定的路径
#   2. 本机 autotrader 的实时股池（本地运行时可用）
#   3. 仓库内提交的快照 universe_ai.json（GitHub Actions 云端运行用）
# 全部缺失或股池超过 21 天未更新 → 降级到全市场成交额筛选（①b）。

AI_UNIVERSE_MAX_AGE_DAYS = 21
AI_UNIVERSE_PATHS = [
    os.environ.get("UNIVERSE_FILE", ""),
    os.path.expanduser(
        "~/Documents/GitHub/deer-flow/autotrader/research/universe.json"),
    os.path.join(os.path.dirname(__file__), "universe_ai.json"),
]


def load_ai_universe() -> tuple[list[str] | None, str]:
    """返回 (股池, 说明)；不可用时返回 (None, 原因)。"""
    for path in AI_UNIVERSE_PATHS:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                uni = json.load(f)
            symbols = uni.get("all_tickers") or []
            generated = date.fromisoformat(uni["generated_at"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"   ⚠️ AI股池文件损坏 {os.path.basename(path)}: {e}")
            continue
        if not symbols:
            continue
        age = (date.today() - generated).days
        if age > AI_UNIVERSE_MAX_AGE_DAYS:
            return None, (f"AI股池已{age}天未更新(>{AI_UNIVERSE_MAX_AGE_DAYS}天上限) "
                          f"@ {os.path.basename(path)}")
        return symbols, (f"AI主题股池 {uni['generated_at']}（{age}天前生成，"
                         f"{len(symbols)}只）@ {os.path.basename(path)}")
    return None, "未找到可用的AI股池文件"


def get_watchlist(trading_client: TradingClient) -> list[str]:
    """分层取股池：AI 主题股池 → 全市场成交额筛选 → 固定列表。"""
    symbols, note = load_ai_universe()
    if symbols:
        print(f"   ✅ {note}")
        log_decision("UNIVERSE", "-", "ai-thematic", note)
        return symbols
    print(f"   ⚠️ {note} → 降级：全市场成交额筛选")
    log_decision("UNIVERSE", "-", "volume-fallback", note)
    return build_universe(trading_client)


# ─────────────────────────────────────────────
# ①b 调研员：全市场动态选股（AI股池不可用时的兜底）
# ─────────────────────────────────────────────

def build_universe(trading_client: TradingClient) -> list[str]:
    """从全市场可交易美股中按成交额筛出前 UNIVERSE_SIZE 只。

    两段式：先拉全部活跃资产（~1万只），再用行情快照按
    当日成交额（价格×成交量）排序取头部。结果按日缓存。
    """
    # 当天已经筛过就直接用缓存（launchd重跑/手动重跑不重复扫描）
    if os.path.exists(UNIVERSE_CACHE):
        try:
            with open(UNIVERSE_CACHE) as f:
                cached = json.load(f)
            if cached.get("date") == date.today().isoformat() and cached.get("symbols"):
                print(f"   （使用今日缓存的股池，{len(cached['symbols'])}只）")
                return cached["symbols"]
        except (json.JSONDecodeError, KeyError):
            pass

    print("   拉取全市场资产列表 ...")
    assets = trading_client.get_all_assets(GetAssetsRequest(
        status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    # 排除ETF/基金：打分逻辑依赖个股基本面（PE/利润率/增长），
    # ETF没有这些数据，只会白占股池名额。Alpaca无ETF标志位，按名称关键词过滤。
    ETF_WORDS = ("ETF", "ETN", "FUND", "TRUST", "SHARES", "INDEX", "BULL", "BEAR")
    symbols = [a.symbol for a in assets
               if a.tradable and a.fractionable
               and a.symbol.isalpha() and len(a.symbol) <= 5
               and not any(w in (a.name or "").upper() for w in ETF_WORDS)]
    print(f"   可交易+支持碎股的普通股（已排除ETF）：{len(symbols)}只，按成交额排序中 ...")

    data_client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    ranked = []
    CHUNK = 500
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        try:
            snaps = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=chunk))
        except Exception as e:
            print(f"   ⚠️ 快照批次 {i//CHUNK + 1} 失败: {e}")
            continue
        for sym, snap in snaps.items():
            bar = snap.daily_bar or snap.previous_daily_bar
            if bar is None or bar.close is None or bar.volume is None:
                continue
            if bar.close < MIN_PRICE:
                continue
            ranked.append((bar.close * bar.volume, sym))

    if len(ranked) < UNIVERSE_SIZE:
        print(f"   ⚠️ 有效快照仅{len(ranked)}只，不足以筛选 → 回退到固定观察列表")
        return FALLBACK_WATCHLIST

    ranked.sort(reverse=True)
    universe = [sym for _, sym in ranked[:UNIVERSE_SIZE]]
    with open(UNIVERSE_CACHE, "w") as f:
        json.dump({"date": date.today().isoformat(), "symbols": universe}, f)
    print(f"   ✅ 今日股池：全市场成交额前{UNIVERSE_SIZE}只（已缓存）")
    return universe


# ─────────────────────────────────────────────
# ⑤ 记录员
# ─────────────────────────────────────────────

def log_decision(action: str, symbol: str, reason: str, detail: str = ""):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "action", "symbol", "reason", "detail"])
        w.writerow([datetime.now(timezone.utc).isoformat(), action, symbol, reason, detail])


# ─────────────────────────────────────────────
# ③+④ 风控官 + 执行员
# ─────────────────────────────────────────────

def get_client() -> TradingClient:
    return TradingClient(os.environ["ALPACA_API_KEY"],
                         os.environ["ALPACA_SECRET_KEY"], paper=True)


def count_today_buys(client) -> int:
    req = GetOrdersRequest(status=QueryOrderStatus.ALL,
                           after=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0))
    orders = client.get_orders(req)
    return sum(1 for o in orders if o.side == OrderSide.BUY)


def run_cycle():
    client = get_client()
    acct = client.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)
    positions = {p.symbol: p for p in client.get_all_positions()}
    exposure = sum(float(p.market_value) for p in positions.values()) / equity

    print("=" * 64)
    print(f"  Trading Agent 决策循环  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  组合 ${equity:,.0f} | 现金 ${cash:,.0f} | 仓位 {exposure:.0%} | 持仓 {len(positions)}只")
    print("=" * 64)

    # ── 第一步：检查持仓（止损和信号退出优先于一切） ──
    print("\n① 检查现有持仓 ...")
    for sym, pos in positions.items():
        entry = float(pos.avg_entry_price)
        now = float(pos.current_price)
        pnl_pct = now / entry - 1

        if pnl_pct < -STOP_LOSS_PCT:
            print(f"   🛑 {sym} 亏损{pnl_pct:.1%} 触发止损 → 卖出全部")
            client.submit_order(MarketOrderRequest(
                symbol=sym, qty=pos.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            log_decision("SELL(stop)", sym, f"止损:{pnl_pct:.1%}", f"entry={entry} now={now}")
            continue

        try:
            score, detail = score_stock(sym)
        except Exception as e:
            # 数据抖动时宁可按兵不动，也不能因打分失败误卖持仓
            print(f"   ⚠️ {sym} 打分失败（{e}），本轮按持有处理")
            log_decision("HOLD(data-error)", sym, str(e)[:100], "")
            continue
        if score <= SELL_THRESHOLD:
            print(f"   📉 {sym} 综合分{score:+.0f} ≤ {SELL_THRESHOLD} → 信号退出，卖出")
            client.submit_order(MarketOrderRequest(
                symbol=sym, qty=pos.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            log_decision("SELL(signal)", sym, f"score={score:+.0f}", detail)
        else:
            print(f"   ✅ {sym} 综合分{score:+.0f}，盈亏{pnl_pct:+.1%}，继续持有")
            log_decision("HOLD", sym, f"score={score:+.0f} pnl={pnl_pct:+.1%}", detail)

    # ── 第二步：构建股池（AI主题股池优先） + 扫描找买入机会 ──
    print("\n② 构建今日股池 ...")
    watchlist = get_watchlist(client)
    print(f"\n   扫描 {len(watchlist)} 只股票 ...")
    todays_buys = count_today_buys(client)
    candidates = []
    for sym in watchlist:
        if sym in positions:
            continue
        try:
            score, detail = score_stock(sym)
        except Exception as e:
            print(f"   ⚠️ {sym} 数据获取失败: {e}")
            continue
        flag = "→ 候选" if score >= BUY_THRESHOLD else ""
        print(f"   {sym:<6} 综合分 {score:+.0f}  {detail}  {flag}")
        if score >= BUY_THRESHOLD:
            candidates.append((score, sym, detail))

    # ── 第三步：风控审批 + 执行（分数最高的优先） ──
    print("\n③ 风控审批 ...")
    candidates.sort(reverse=True)
    bought = 0
    for score, sym, detail in candidates:
        if todays_buys + bought >= MAX_NEW_BUYS_A_DAY:
            print(f"   ⛔ 已达单日新开仓上限({MAX_NEW_BUYS_A_DAY})，{sym} 留待明天")
            log_decision("SKIP(day-cap)", sym, f"score={score:+.0f}", detail)
            continue
        if exposure >= MAX_EXPOSURE_PCT:
            print(f"   ⛔ 总仓位{exposure:.0%}已达上限，跳过 {sym}")
            log_decision("SKIP(exposure)", sym, f"score={score:+.0f}", detail)
            continue
        size = min(equity * ENTRY_SIZE_PCT, equity * MAX_POSITION_PCT, cash * 0.95)
        if size < 100:
            print(f"   ⛔ 可用现金不足，跳过 {sym}")
            continue
        print(f"   ✅ 买入 {sym}：${size:,.0f}（综合分{score:+.0f}）")
        client.submit_order(MarketOrderRequest(
            symbol=sym, notional=round(size, 2), side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
        log_decision("BUY", sym, f"score={score:+.0f} size=${size:,.0f}", detail)
        bought += 1
        exposure += size / equity
        cash -= size

    if not candidates:
        print("   （无股票达到买入阈值——空仓等待也是决策）")

    print("\n完成。决策日志 → decisions_log.csv")
    print("提醒：美股休市时段提交的订单会排队到下一个开盘时刻成交。")


def show_status():
    client = get_client()
    acct = client.get_account()
    equity = float(acct.equity)
    print(f"组合总值 ${equity:,.2f} | 现金 ${float(acct.cash):,.2f}")
    positions = client.get_all_positions()
    if not positions:
        print("当前无持仓")
        return
    print(f"\n{'股票':<8}{'数量':>10}{'成本':>10}{'现价':>10}{'盈亏':>10}")
    for p in positions:
        pnl = float(p.unrealized_plpc)
        print(f"{p.symbol:<8}{float(p.qty):>10.2f}{float(p.avg_entry_price):>10.2f}"
              f"{float(p.current_price):>10.2f}{pnl:>10.1%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_status()
    elif len(sys.argv) > 1 and sys.argv[1] == "universe":
        # 只打印今日实际会使用的股池，不打分不交易（用于验证）
        syms = get_watchlist(get_client())
        print(f"\n今日股池（{len(syms)}只）：")
        for i in range(0, len(syms), 10):
            print("  " + " ".join(f"{s:<6}" for s in syms[i:i+10]))
    else:
        run_cycle()
