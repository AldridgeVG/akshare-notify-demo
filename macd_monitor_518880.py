import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import akshare as ak
import pandas as pd
import numpy as np

# ===================== 配置 =====================
STOCK_CODE = "518880"          # 华安黄金ETF（上交所）
STOCK_NAME = "黄金ETF"          # 用于日志显示
POLL_INTERVAL = 60               # 轮询间隔（秒）
HISTORY_DAYS = 120             # 历史日K线获取天数

# MACD 参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# KDJ 参数
KDJ_N = 9
KDJ_M1 = 3   # K 平滑系数
KDJ_M2 = 3   # D 平滑系数

# 成交量参数
VOL_MA_DAYS = 20               # 成交量均线周期
VOL_RATIO_THRESHOLD = 1.50   # 量比 > 1.5 视为显著放量
VOL_RATIO_SHRINK = 0.60      # 量比 < 0.6 视为显著缩量

# 显著红绿柱判定阈值
BAR_SIGNIFICANT_RATIO = 1.30   # 当前柱体较近期同色系平均柱体放大30%视为显著
BAR_RECENT_N = 5               # 近期参考柱体数量

# 背离检测参数
DEV_LOOKBACK = 30              # 背离检测回溯天数
PEAK_MIN_DISTANCE = 5          # 高低点最小距离

# 综合评分阈值
SCORE_STRONG_BUY = 3           # 强烈买入评分阈值
SCORE_BUY = 2                  # 买入评分阈值
SCORE_SELL = -2                # 卖出评分阈值
SCORE_STRONG_SELL = -3         # 强烈卖出评分阈值

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"macd_monitor_{STOCK_CODE}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MACD_Monitor")


# ===================== 技术指标计算 =====================
def compute_macd(close_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    计算 MACD 指标。
    返回 DataFrame，包含 close, DIF, DEA, MACD_HIST
    """
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return pd.DataFrame({
        "close": close_series,
        "DIF": dif,
        "DEA": dea,
        "MACD_HIST": macd_hist
    })


def compute_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """
    计算 KDJ 指标。
    
    参数:
        df: DataFrame 包含 high, low, close 列
        n: RSV 计算周期（默认9）
        m1: K 平滑因子（默认3）
        m2: D 平滑因子（默认3）
    
    返回:
        DataFrame 包含 K, D, J
    """
    low_min = df["low"].rolling(window=n, min_periods=n).min()
    high_max = df["high"].rolling(window=n, min_periods=n).max()
    
    # RSV（未成熟随机值）
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    
    # K 值: 当日K = 2/3 * 前一日K + 1/3 * 当日RSV
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    
    # D 值: 当日D = 2/3 * 前一日D + 1/3 * 当日K
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    
    # J 值: J = 3K - 2D
    j = 3 * k - 2 * d
    
    return pd.DataFrame({
        "K": k,
        "D": d,
        "J": j
    })


def compute_volume_signals(df: pd.DataFrame, ma_days: int = VOL_MA_DAYS) -> pd.DataFrame:
    """
    计算成交量相关信号。
    
    返回 DataFrame 包含:
        - VOL_MA: 成交量均线
        - VOL_RATIO: 量比（当前成交量 / 均线）
        - VOL_SIGNAL: 成交量信号描述
    """
    vol_ma = df["volume"].rolling(window=ma_days, min_periods=ma_days).mean()
    vol_ratio = df["volume"] / vol_ma
    
    return pd.DataFrame({
        "VOL_MA": vol_ma,
        "VOL_RATIO": vol_ratio
    })


# ===================== 数据获取 =====================
def fetch_history_daily(symbol: str = STOCK_CODE, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """
    获取ETF历史日K线数据（含 open/high/low/close/volume）。
    使用 ak.fund_etf_hist_em（东财ETF数据源）。
    """
    logger.info(f"正在获取 {symbol} 历史日K线数据...")
    
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
    except Exception as e:
        logger.warning(f"fund_etf_hist_em 带日期参数失败: {e}，尝试获取全部历史...")
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
        df = df.tail(days).copy()
    
    df = df.rename(columns={
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    
    # 确保数值类型
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    logger.info(f"历史数据获取完成，共 {len(df)} 条记录，最新收盘: {df['close'].iloc[-1]:.3f}")
    return df


def fetch_spot_price(symbol: str = STOCK_CODE) -> Optional[Tuple[float, str, str]]:
    """
    获取ETF实时行情快照。
    返回: (最新价, 数据日期, 更新时间) 或 None
    """
    try:
        df = ak.fund_etf_spot_em()
        row = df[df["代码"] == symbol]
        if row.empty:
            logger.warning(f"未在 fund_etf_spot_em 中查到 {symbol}")
            return None
        
        latest_price = float(row["最新价"].iloc[0])
        
        date_raw = row["数据日期"].iloc[0] if "数据日期" in row.columns else ""
        time_raw = row["更新时间"].iloc[0] if "更新时间" in row.columns else ""
        
        try:
            date_str = pd.to_datetime(date_raw).strftime("%Y-%m-%d") if date_raw else ""
        except Exception:
            date_str = str(date_raw)[:10]
        
        try:
            time_str = pd.to_datetime(time_raw).strftime("%H:%M:%S") if time_raw else ""
        except Exception:
            time_s = str(time_raw)
            if " " in time_s:
                time_str = time_s.split(" ")[-1][:8]
            else:
                time_str = time_s[:8]
        
        return latest_price, date_str, time_str
    except Exception as e:
        logger.error(f"获取实时行情失败: {e}")
        return None


# ===================== MACD 信号检测 =====================
def detect_significant_bar(macd_df: pd.DataFrame, lookback: int = BAR_RECENT_N, ratio: float = BAR_SIGNIFICANT_RATIO) -> str:
    """
    检测"显著红绿柱"。
    """
    hist = macd_df["MACD_HIST"]
    if len(hist) < lookback + 2:
        return ""
    
    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    
    same_color_bars = []
    for i in range(2, lookback + 2):
        val = hist.iloc[-i]
        if curr > 0 and val > 0:
            same_color_bars.append(val)
        elif curr < 0 and val < 0:
            same_color_bars.append(abs(val))
    
    if not same_color_bars:
        return ""
    
    if curr > 0:
        avg_bar = np.mean(same_color_bars)
        max_bar = max(same_color_bars)
        if curr >= max_bar or (prev > 0 and curr >= prev * ratio) or curr >= avg_bar * ratio:
            return f"显著红柱(柱高={curr:.4f}, 近期均值={avg_bar:.4f})"
    else:
        curr_abs = abs(curr)
        avg_bar = np.mean(same_color_bars)
        max_bar = max(same_color_bars)
        if curr_abs >= max_bar or (prev < 0 and curr_abs >= abs(prev) * ratio) or curr_abs >= avg_bar * ratio:
            return f"显著绿柱(柱高={curr:.4f}, 近期均值=-{avg_bar:.4f})"
    
    return ""


def find_peaks_and_troughs(series: pd.Series, min_distance: int = PEAK_MIN_DISTANCE) -> Tuple[List[int], List[int]]:
    """
    局部高低点检测。
    """
    peaks = []
    troughs = []
    n = len(series)
    for i in range(min_distance, n - min_distance):
        window = series.iloc[i - min_distance: i + min_distance + 1]
        if series.iloc[i] == window.max():
            peaks.append(i)
        elif series.iloc[i] == window.min():
            troughs.append(i)
    return peaks, troughs


def detect_divergence(macd_df: pd.DataFrame, lookback: int = DEV_LOOKBACK) -> str:
    """
    检测MACD背离信号。
    """
    if len(macd_df) < lookback:
        lookback = len(macd_df) - 5
    
    sub_df = macd_df.iloc[-lookback:].copy()
    close = sub_df["close"]
    hist = sub_df["MACD_HIST"]
    dif = sub_df["DIF"]
    
    peaks, troughs = find_peaks_and_troughs(close, min_distance=PEAK_MIN_DISTANCE)
    
    signal = ""
    
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        price1, price2 = close.iloc[p1], close.iloc[p2]
        hist1, hist2 = hist.iloc[p1], hist.iloc[p2]
        dif1, dif2 = dif.iloc[p1], dif.iloc[p2]
        
        if price2 > price1 and (hist2 < hist1 or dif2 < dif1):
            signal = f"MACD顶背离(价:{price1:.3f}->{price2:.3f}, HIST:{hist1:.4f}->{hist2:.4f})【建议减仓/观望】"
    
    if not signal and len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        price1, price2 = close.iloc[t1], close.iloc[t2]
        hist1, hist2 = hist.iloc[t1], hist.iloc[t2]
        dif1, dif2 = dif.iloc[t1], dif.iloc[t2]
        
        if price2 < price1 and (hist2 > hist1 or dif2 > dif1):
            signal = f"MACD底背离(价:{price1:.3f}->{price2:.3f}, HIST:{hist1:.4f}->{hist2:.4f})【建议买进/加仓】"
    
    return signal


# ===================== KDJ 信号检测 =====================
def detect_kdj_signals(kdj_df: pd.DataFrame, price_df: pd.DataFrame) -> Tuple[str, str]:
    """
    检测 KDJ 买卖信号。
    
    返回: (kdj_signal, kdj_divergence)
        - kdj_signal: 当前KDJ状态描述（金叉/死叉/超买/超卖等）
        - kdj_divergence: KDJ背离信号
    """
    if len(kdj_df) < 2:
        return "", ""
    
    latest = kdj_df.iloc[-1]
    prev = kdj_df.iloc[-2]
    k, d, j = latest["K"], latest["D"], latest["J"]
    prev_k, prev_d = prev["K"], prev["D"]
    
    signals = []
    
    # 金叉/死叉
    if prev_k < prev_d and k > d:
        if k < 30:
            signals.append(f"KDJ低位金叉(K={k:.2f}, D={d:.2f}, J={j:.2f})【强买入信号】")
        else:
            signals.append(f"KDJ金叉(K={k:.2f}, D={d:.2f}, J={j:.2f})")
    elif prev_k > prev_d and k < d:
        if k > 70:
            signals.append(f"KDJ高位死叉(K={k:.2f}, D={d:.2f}, J={j:.2f})【强卖出信号】")
        else:
            signals.append(f"KDJ死叉(K={k:.2f}, D={d:.2f}, J={j:.2f})")
    
    # 超买/超卖
    if j > 100:
        signals.append(f"KDJ超买区(J={j:.2f})")
    elif j < 0:
        signals.append(f"KDJ超卖区(J={j:.2f})")
    elif k > 80 and d > 80:
        signals.append(f"KDJ高位运行(K={k:.2f}, D={d:.2f})")
    elif k < 20 and d < 20:
        signals.append(f"KDJ低位运行(K={k:.2f}, D={d:.2f})")
    
    kdj_signal = "; ".join(signals) if signals else f"KDJ中性(K={k:.2f}, D={d:.2f}, J={j:.2f})"
    
    # KDJ背离检测（使用K值或D值与价格比较）
    kdj_div = detect_kdj_divergence(price_df, kdj_df)
    
    return kdj_signal, kdj_div


def detect_kdj_divergence(price_df: pd.DataFrame, kdj_df: pd.DataFrame, lookback: int = DEV_LOOKBACK) -> str:
    """
    检测KDJ背离：价格创新高/低但KDJ未同步。
    """
    if len(price_df) < lookback or len(kdj_df) < lookback:
        lookback = min(len(price_df), len(kdj_df)) - 5
    
    close = price_df["close"].iloc[-lookback:]
    k_values = kdj_df["K"].iloc[-lookback:]
    
    peaks, troughs = find_peaks_and_troughs(close, min_distance=PEAK_MIN_DISTANCE)
    
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        price1, price2 = close.iloc[p1], close.iloc[p2]
        k1, k2 = k_values.iloc[p1], k_values.iloc[p2]
        if price2 > price1 and k2 < k1:
            return f"KDJ顶背离(价:{price1:.3f}->{price2:.3f}, K:{k1:.2f}->{k2:.2f})【减仓警示】"
    
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        price1, price2 = close.iloc[t1], close.iloc[t2]
        k1, k2 = k_values.iloc[t1], k_values.iloc[t2]
        if price2 < price1 and k2 > k1:
            return f"KDJ底背离(价:{price1:.3f}->{price2:.3f}, K:{k1:.2f}->{k2:.2f})【买入机会】"
    
    return ""


# ===================== 成交量信号检测 =====================
def detect_volume_signals(df: pd.DataFrame, vol_info: pd.DataFrame) -> str:
    """
    检测成交量信号，结合价格走势判断量价配合。
    
    返回成交量信号描述字符串。
    """
    if len(df) < 2 or len(vol_info) < 1:
        return ""
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    curr_price = latest["close"]
    prev_price = prev["close"]
    curr_vol = latest["volume"]
    vol_ratio = vol_info["VOL_RATIO"].iloc[-1] if not pd.isna(vol_info["VOL_RATIO"].iloc[-1]) else 1.0
    
    price_change = (curr_price - prev_price) / prev_price * 100 if prev_price != 0 else 0
    
    signals = []
    
    # 放量判断
    if vol_ratio > VOL_RATIO_THRESHOLD:
        if price_change > 1.0:
            signals.append(f"放量上涨(量比{vol_ratio:.2f}, 涨幅{price_change:.2f}%)【多头资金进场】")
        elif price_change < -1.0:
            signals.append(f"放量下跌(量比{vol_ratio:.2f}, 跌幅{abs(price_change):.2f}%)【空头抛压加重】")
        else:
            signals.append(f"显著放量(量比{vol_ratio:.2f})【注意变盘】")
    elif vol_ratio < VOL_RATIO_SHRINK:
        if abs(price_change) < 0.5:
            signals.append(f"明显缩量盘整(量比{vol_ratio:.2f})【观望】")
        elif price_change > 0:
            signals.append(f"缩量上涨(量比{vol_ratio:.2f})【上涨动能不足】")
        else:
            signals.append(f"缩量下跌(量比{vol_ratio:.2f})【抛压减轻/可能止跌】")
    else:
        signals.append(f"成交量正常(量比{vol_ratio:.2f})")
    
    return "; ".join(signals)


# ===================== 综合评分与建议 =====================
def calculate_comprehensive_score(
    macd_df: pd.DataFrame,
    bar_signal: str,
    div_signal: str,
    kdj_df: pd.DataFrame,
    kdj_signal: str,
    kdj_div: str,
    vol_signal: str
) -> Tuple[int, str]:
    """
    多指标综合评分系统。
    
    评分规则:
        买入加分项:
            +1: MACD金叉 / 显著红柱
            +1: MACD底背离
            +1: KDJ低位金叉(<30) / KDJ底背离 / J<0超卖
            +1: 放量上涨(量比>1.5且涨幅>1%)
            +1: 缩量止跌(量比<0.6且价格微跌或持平)
        
        卖出减分项:
            -1: MACD死叉 / 显著绿柱
            -1: MACD顶背离
            -1: KDJ高位死叉(>70) / KDJ顶背离 / J>100超买
            -1: 放量下跌(量比>1.5且跌幅>1%)
            -1: 缩量上涨(量比<0.6且价格上涨)
    
    返回: (score, score_desc)
    """
    score = 0
    reasons = []
    
    # --- MACD 评分 ---
    if len(macd_df) >= 2:
        latest = macd_df.iloc[-1]
        prev = macd_df.iloc[-2]
        
        # 金叉/死叉
        if prev["DIF"] < prev["DEA"] and latest["DIF"] > latest["DEA"]:
            score += 1
            reasons.append("MACD金叉(+1)")
        elif prev["DIF"] > prev["DEA"] and latest["DIF"] < latest["DEA"]:
            score -= 1
            reasons.append("MACD死叉(-1)")
        
        # 显著柱体
        if "显著红柱" in bar_signal:
            score += 1
            reasons.append("MACD显著红柱(+1)")
        elif "显著绿柱" in bar_signal:
            score -= 1
            reasons.append("MACD显著绿柱(-1)")
        
        # 背离
        if "底背离" in div_signal:
            score += 1
            reasons.append("MACD底背离(+1)")
        elif "顶背离" in div_signal:
            score -= 1
            reasons.append("MACD顶背离(-1)")
    
    # --- KDJ 评分 ---
    if len(kdj_df) >= 2:
        latest_kdj = kdj_df.iloc[-1]
        prev_kdj = kdj_df.iloc[-2]
        k, d, j = latest_kdj["K"], latest_kdj["D"], latest_kdj["J"]
        prev_k = prev_kdj["K"]
        
        # 金叉/死叉
        if prev_k < prev_kdj["D"] and k > d:
            if k < 30:
                score += 2
                reasons.append("KDJ低位金叉(+2)")
            else:
                score += 1
                reasons.append("KDJ金叉(+1)")
        elif prev_k > prev_kdj["D"] and k < d:
            if k > 70:
                score -= 2
                reasons.append("KDJ高位死叉(-2)")
            else:
                score -= 1
                reasons.append("KDJ死叉(-1)")
        
        # 超买超卖
        if j > 100:
            score -= 1
            reasons.append("KDJ超买(J>100)(-1)")
        elif j < 0:
            score += 1
            reasons.append("KDJ超卖(J<0)(+1)")
        
        # 背离
        if "底背离" in kdj_div:
            score += 1
            reasons.append("KDJ底背离(+1)")
        elif "顶背离" in kdj_div:
            score -= 1
            reasons.append("KDJ顶背离(-1)")
    
    # --- 成交量评分 ---
    if "放量上涨" in vol_signal:
        score += 1
        reasons.append("放量上涨(+1)")
    elif "放量下跌" in vol_signal:
        score -= 1
        reasons.append("放量下跌(-1)")
    elif "缩量上涨" in vol_signal:
        score -= 1
        reasons.append("缩量上涨(-1)")
    elif "缩量下跌" in vol_signal:
        score += 1
        reasons.append("缩量下跌/抛压减轻(+1)")
    
    # 评分描述
    if score >= SCORE_STRONG_BUY:
        score_desc = f"强烈看多(评分:{score})"
    elif score >= SCORE_BUY:
        score_desc = f"偏多/买入(评分:{score})"
    elif score <= SCORE_STRONG_SELL:
        score_desc = f"强烈看空(评分:{score})"
    elif score <= SCORE_SELL:
        score_desc = f"偏空/卖出(评分:{score})"
    else:
        score_desc = f"中性观望(评分:{score})"
    
    return score, score_desc, reasons


def generate_comprehensive_advice(score: int, score_desc: str, reasons: List[str]) -> str:
    """
    根据综合评分生成最终操作建议。
    """
    advice = f"【{score_desc}】"
    
    if score >= SCORE_STRONG_BUY:
        advice += " 多指标共振，买入信号强烈，建议积极建仓/加仓。"
    elif score >= SCORE_BUY:
        advice += " 多头信号占优，建议逢低买入或持仓待涨。"
    elif score <= SCORE_STRONG_SELL:
        advice += " 多指标共振，卖出信号强烈，建议果断减仓/清仓。"
    elif score <= SCORE_SELL:
        advice += " 空头信号占优，建议减仓避险或空仓观望。"
    else:
        advice += " 多空信号交织，方向不明，建议观望等待明确信号。"
    
    if reasons:
        advice += f" 依据: {', '.join(reasons)}"
    
    return advice


# ===================== 主监控循环 =====================
def main():
    logger.info("=" * 60)
    logger.info(f"多指标综合监控系统启动 | 标的: {STOCK_CODE}({STOCK_NAME})")
    logger.info(f"AKShare 版本: {ak.__version__}")
    logger.info("指标: MACD + KDJ + 成交量")
    logger.info("=" * 60)
    
    # 1. 初始化历史数据
    try:
        hist_df = fetch_history_daily(STOCK_CODE, HISTORY_DAYS)
        
        # 计算 MACD
        macd_df = compute_macd(hist_df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        
        # 计算 KDJ
        kdj_df = compute_kdj(hist_df, KDJ_N, KDJ_M1, KDJ_M2)
        
        # 计算成交量信号
        vol_info = compute_volume_signals(hist_df, VOL_MA_DAYS)
        
        # 合并所有指标
        hist_df = hist_df.join(macd_df[["DIF", "DEA", "MACD_HIST"]])
        hist_df = hist_df.join(kdj_df[["K", "D", "J"]])
        hist_df = hist_df.join(vol_info[["VOL_MA", "VOL_RATIO"]])
        
    except Exception as e:
        logger.error(f"初始化历史数据失败: {e}")
        sys.exit(1)
    
    logger.info(f"初始化完成，已加载 {len(hist_df)} 根日K线，所有指标计算就绪。")
    
    # 2. 主循环
    last_date = hist_df.index[-1].date()
    loop_count = 0
    
    while True:
        loop_count += 1
        now = datetime.now()
        
        try:
            # 获取实时价格
            spot = fetch_spot_price(STOCK_CODE)
            if spot is None:
                logger.warning("本次轮询未获取到实时价格，跳过...")
                time.sleep(POLL_INTERVAL)
                continue
            
            latest_price, data_date_str, data_time_str = spot
            logger.info(f"[{loop_count}] 实时价格: {latest_price:.3f} | 数据时间: {data_date_str} {data_time_str}")
            
            # 判断是否需要更新历史数据（跨日时）
            today = now.date()
            if today > last_date:
                logger.info("检测到新交易日，重新拉取历史数据...")
                try:
                    new_hist_df = fetch_history_daily(STOCK_CODE, HISTORY_DAYS)
                    new_macd_df = compute_macd(new_hist_df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
                    new_kdj_df = compute_kdj(new_hist_df, KDJ_N, KDJ_M1, KDJ_M2)
                    new_vol_info = compute_volume_signals(new_hist_df, VOL_MA_DAYS)
                    
                    hist_df = new_hist_df.join(new_macd_df[["DIF", "DEA", "MACD_HIST"]])
                    hist_df = hist_df.join(new_kdj_df[["K", "D", "J"]])
                    hist_df = hist_df.join(new_vol_info[["VOL_MA", "VOL_RATIO"]])
                    
                    last_date = today
                    logger.info("历史数据已刷新")
                except Exception as e:
                    logger.warning(f"刷新历史数据失败，继续使用旧数据: {e}")
            
            # --- 更新今日数据（用实时价格替代当日收盘价进行近似估算） ---
            updated_hist = hist_df.copy()
            updated_hist.loc[updated_hist.index[-1], "close"] = latest_price
            
            # 重新计算所有指标
            updated_macd = compute_macd(updated_hist["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            updated_kdj = compute_kdj(updated_hist, KDJ_N, KDJ_M1, KDJ_M2)
            updated_vol = compute_volume_signals(updated_hist, VOL_MA_DAYS)
            
            # --- 检测各指标信号 ---
            # MACD
            bar_signal = detect_significant_bar(updated_macd)
            div_signal = detect_divergence(updated_macd)
            
            # KDJ
            kdj_signal, kdj_div = detect_kdj_signals(updated_kdj, updated_hist)
            
            # 成交量
            vol_signal = detect_volume_signals(updated_hist, updated_vol)
            
            # 综合评分
            score, score_desc, reasons = calculate_comprehensive_score(
                updated_macd, bar_signal, div_signal,
                updated_kdj, kdj_signal, kdj_div,
                vol_signal
            )
            advice = generate_comprehensive_advice(score, score_desc, reasons)
            
            # --- 获取最新指标值 ---
            latest_macd = updated_macd.iloc[-1]
            latest_kdj = updated_kdj.iloc[-1]
            latest_vol = updated_vol.iloc[-1]
            
            # --- 打印综合日志 ---
            logger.info(
                f"MACD | DIF={latest_macd['DIF']:.4f} DEA={latest_macd['DEA']:.4f} "
                f"HIST={latest_macd['MACD_HIST']:.4f}"
            )
            logger.info(
                f"KDJ  | K={latest_kdj['K']:.2f} D={latest_kdj['D']:.2f} J={latest_kdj['J']:.2f}"
            )
            logger.info(
                f"VOL  | 量比={latest_vol['VOL_RATIO']:.2f} 均量={latest_vol['VOL_MA']:,.0f} "
                f"现量={updated_hist['volume'].iloc[-1]:,.0f}"
            )
            
            if bar_signal:
                logger.info(f"【MACD柱体】{bar_signal}")
            if div_signal:
                logger.info(f"【MACD背离】{div_signal}")
            if kdj_signal and "中性" not in kdj_signal:
                logger.info(f"【KDJ信号】{kdj_signal}")
            if kdj_div:
                logger.info(f"【KDJ背离】{kdj_div}")
            if vol_signal:
                logger.info(f"【成交信号】{vol_signal}")
            
            logger.info(f"【综合评分】{score_desc}")
            logger.info(f"【操作建议】{advice}")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"主循环异常: {e}", exc_info=True)
        
        # 等待下一轮
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("监控程序被用户中断，退出。")
