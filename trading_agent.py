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
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

# 复用双分析师的打分逻辑（同目录的 analyst_v2.py）
from analyst_v2 import (fetch_all, t_trend, t_rsi, t_macd, t_volume,
                        f_valuation, f_profitability, f_growth, f_balance)

# ─────────────────────────────────────────────
# 配置（观察列表和阈值可调；风控规则不建议放松）
# ─────────────────────────────────────────────

WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
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

        score, detail = score_stock(sym)
        if score <= SELL_THRESHOLD:
            print(f"   📉 {sym} 综合分{score:+.0f} ≤ {SELL_THRESHOLD} → 信号退出，卖出")
            client.submit_order(MarketOrderRequest(
                symbol=sym, qty=pos.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            log_decision("SELL(signal)", sym, f"score={score:+.0f}", detail)
        else:
            print(f"   ✅ {sym} 综合分{score:+.0f}，盈亏{pnl_pct:+.1%}，继续持有")
            log_decision("HOLD", sym, f"score={score:+.0f} pnl={pnl_pct:+.1%}", detail)

    # ── 第二步：扫描观察列表找买入机会 ──
    print("\n② 扫描观察列表 ...")
    todays_buys = count_today_buys(client)
    candidates = []
    for sym in WATCHLIST:
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
    else:
        run_cycle()
