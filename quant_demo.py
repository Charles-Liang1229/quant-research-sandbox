"""
量化AI Demo — XGBoost 预测A股次日涨跌方向 + 简单回测
标的：沪深300成分股之一（平安银行 000001）
时间：近3年日线数据
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. 数据获取
# ─────────────────────────────────────────────

def fetch_data(symbol: str = "AAPL", start: str = "2022-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    print(f"📥 下载 {symbol} 日线数据 {start}~{end} ...")
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.astype(float)
    df["pct_chg"] = df["close"].pct_change() * 100
    print(f"   获取 {len(df)} 条记录\n")
    return df


# ─────────────────────────────────────────────
# 2. 特征工程
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("⚙️  构建技术指标特征 ...")
    f = df.copy()

    # 收益率特征
    for n in [1, 3, 5, 10, 20]:
        f[f"ret_{n}d"] = f["close"].pct_change(n)

    # 移动均线偏差
    for n in [5, 10, 20, 60]:
        ma = f["close"].rolling(n).mean()
        f[f"ma_dev_{n}"] = (f["close"] - ma) / ma

    # 成交量特征
    for n in [5, 10, 20]:
        f[f"vol_ratio_{n}"] = f["volume"] / f["volume"].rolling(n).mean()

    # RSI
    f["rsi_6"]  = ta.momentum.RSIIndicator(f["close"], window=6).rsi()
    f["rsi_14"] = ta.momentum.RSIIndicator(f["close"], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(f["close"])
    f["macd"]        = macd.macd()
    f["macd_signal"] = macd.macd_signal()
    f["macd_diff"]   = macd.macd_diff()

    # Bollinger Band 位置
    bb = ta.volatility.BollingerBands(f["close"], window=20)
    f["bb_pct"] = bb.bollinger_pband()

    # ATR 波动率
    f["atr_14"] = ta.volatility.AverageTrueRange(f["high"], f["low"], f["close"], window=14).average_true_range()
    f["atr_pct"] = f["atr_14"] / f["close"]

    # CCI
    f["cci_14"] = ta.trend.CCIIndicator(f["high"], f["low"], f["close"], window=14).cci()

    # 标签：明天是否上涨（严格shift，避免数据泄露）
    f["label"] = (f["close"].shift(-1) > f["close"]).astype(int)

    f.dropna(inplace=True)
    print(f"   特征数量: {len([c for c in f.columns if c != 'label'])} 个，样本数: {len(f)}\n")
    return f


# ─────────────────────────────────────────────
# 3. 训练 / 验证 / 测试 划分（时序切分）
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    feature_cols = [c for c in df.columns if c not in
                    ["label", "open", "high", "low", "close", "volume", "pct_chg"]]

    X = df[feature_cols]
    y = df["label"]

    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": (X.iloc[:n_train],           y.iloc[:n_train]),
        "val":   (X.iloc[n_train:n_train+n_val], y.iloc[n_train:n_train+n_val]),
        "test":  (X.iloc[n_train+n_val:],      y.iloc[n_train+n_val:]),
    }
    print(f"📊 数据划分: 训练={n_train}天 | 验证={n_val}天 | 测试={n - n_train - n_val}天\n")
    return splits, feature_cols


# ─────────────────────────────────────────────
# 4. 模型训练
# ─────────────────────────────────────────────

def train_model(splits):
    print("🤖 训练 XGBoost 模型 ...")
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred  = model.predict(X_val)
    val_acc   = accuracy_score(y_val, val_pred)
    print(f"   验证集准确率: {val_acc:.2%}\n")
    return model


# ─────────────────────────────────────────────
# 5. 测试集评估
# ─────────────────────────────────────────────

def evaluate(model, splits):
    X_test, y_test = splits["test"]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("=" * 50)
    print(f"🎯 测试集准确率: {acc:.2%}")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["下跌", "上涨"]))
    return y_pred


# ─────────────────────────────────────────────
# 6. 简单回测
# ─────────────────────────────────────────────

def backtest(df: pd.DataFrame, splits, y_pred, fee_rate=0.001):
    """
    策略：模型预测上涨则持仓，预测下跌则空仓
    fee_rate: 单边手续费（万1 = 0.001）
    """
    print("\n📈 运行回测 ...")
    X_test, y_test = splits["test"]
    test_df = df.loc[X_test.index].copy()
    test_df["signal"]   = y_pred          # 1=持仓, 0=空仓
    test_df["next_ret"] = test_df["close"].pct_change().shift(-1)

    # 持仓收益（考虑手续费：信号变化时扣费）
    test_df["signal_shift"] = test_df["signal"].shift(1).fillna(0)
    test_df["trade"]        = (test_df["signal"] != test_df["signal_shift"]).astype(int)
    test_df["strat_ret"]    = test_df["signal"] * test_df["next_ret"] - test_df["trade"] * fee_rate

    # 累积收益
    test_df["cum_strat"]  = (1 + test_df["strat_ret"]).cumprod()
    test_df["cum_bh"]     = (1 + test_df["next_ret"]).cumprod()   # 买入持有

    # 指标计算
    strat_annual = test_df["strat_ret"].mean() * 252
    bh_annual    = test_df["next_ret"].mean() * 252
    strat_sharpe = test_df["strat_ret"].mean() / (test_df["strat_ret"].std() + 1e-9) * np.sqrt(252)

    cum = test_df["cum_strat"]
    drawdown = (cum / cum.cummax() - 1)
    max_dd   = drawdown.min()

    print(f"   {'指标':<20} {'策略':>10} {'买入持有':>10}")
    print(f"   {'-'*42}")
    print(f"   {'年化收益':<20} {strat_annual:>9.2%} {bh_annual:>9.2%}")
    print(f"   {'夏普比率':<20} {strat_sharpe:>9.2f}")
    print(f"   {'最大回撤':<20} {max_dd:>9.2%}")
    print(f"   {'总收益':<20} {test_df['cum_strat'].iloc[-2]-1:>9.2%} {test_df['cum_bh'].iloc[-2]-1:>9.2%}")

    return test_df


# ─────────────────────────────────────────────
# 7. 可视化
# ─────────────────────────────────────────────

def plot_results(test_df, feature_cols, model):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("量化AI Demo — XGBoost 预测A股涨跌", fontsize=14, fontweight="bold")

    # (1) 累积收益对比
    ax = axes[0, 0]
    test_df["cum_strat"].plot(ax=ax, label="AI策略", color="#2196F3", linewidth=1.5)
    test_df["cum_bh"].plot(ax=ax, label="买入持有", color="#FF9800", linewidth=1.5, linestyle="--")
    ax.set_title("累积收益对比")
    ax.set_ylabel("累积收益倍数")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) 回撤曲线
    ax = axes[0, 1]
    cum = test_df["cum_strat"]
    drawdown = (cum / cum.cummax() - 1) * 100
    drawdown.plot(ax=ax, color="#F44336", linewidth=1.5)
    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.2, color="#F44336")
    ax.set_title("策略回撤 (%)")
    ax.set_ylabel("回撤 %")
    ax.grid(True, alpha=0.3)

    # (3) 特征重要性 Top 15
    ax = axes[1, 0]
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance.nlargest(15).sort_values().plot(kind="barh", ax=ax, color="#4CAF50", alpha=0.8)
    ax.set_title("特征重要性 Top 15")
    ax.set_xlabel("重要性得分")
    ax.grid(True, alpha=0.3, axis="x")

    # (4) 每日收益分布
    ax = axes[1, 1]
    test_df["strat_ret"].dropna().hist(ax=ax, bins=40, color="#9C27B0", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("每日收益分布")
    ax.set_xlabel("日收益率")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "quant_demo/result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 图表已保存: {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  量化AI Demo  —  XGBoost A股预测回测")
    print("=" * 50 + "\n")

    df = fetch_data(symbol="AAPL", start="2022-01-01", end="2025-01-01")
    df = build_features(df)
    splits, feature_cols = split_data(df)
    model   = train_model(splits)
    y_pred  = evaluate(model, splits)
    test_df = backtest(df, splits, y_pred)
    plot_results(test_df, feature_cols, model)

    print("\n✅ Demo 完成！\n")
    print("下一步建议:")
    print("  1. 换更多标的，做多股票组合")
    print("  2. 加入基本面因子（PE、ROE等）")
    print("  3. 用 Walk-forward 验证避免过拟合")
    print("  4. 尝试 LSTM 替换 XGBoost\n")


if __name__ == "__main__":
    main()
