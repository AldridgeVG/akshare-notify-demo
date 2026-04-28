import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import akshare as ak
import numpy as np
import pandas as pd

from config_loader import CONFIG, STRATEGY

# ===================== 配置 =====================
STOCK_CODE = CONFIG.get("stock", {}).get("code", "518880")
POLL_INTERVAL = CONFIG.get("monitor", {}).get("poll_interval", 60)
HISTORY_DAYS = CONFIG.get("monitor", {}).get("history_days", 120)

# MACD 参数
MACD_FAST = CONFIG.get("indicators", {}).get("macd", {}).get("fast", 12)
MACD_SLOW = CONFIG.get("indicators", {}).get("macd", {}).get("slow", 26)
MACD_SIGNAL = CONFIG.get("indicators", {}).get("macd", {}).get("signal", 9)

# KDJ 参数
KDJ_N = CONFIG.get("indicators", {}).get("kdj", {}).get("n", 9)
KDJ_M1 = CONFIG.get("indicators", {}).get("kdj", {}).get("m1", 3)
KDJ_M2 = CONFIG.get("indicators", {}).get("kdj", {}).get("m2", 3)

# 成交量参数
VOL_MA_DAYS = CONFIG.get("indicators", {}).get("volume", {}).get("ma_days", 20)

# 显著红绿柱判定阈值
BAR_SIGNIFICANT_RATIO = STRATEGY.get("macd_bar", {}).get("significant_ratio", 1.30)
BAR_RECENT_N = STRATEGY.get("macd_bar", {}).get("recent_n", 5)

# 背离检测参数
DEV_LOOKBACK = STRATEGY.get("divergence", {}).get("lookback", 30)
PEAK_MIN_DISTANCE = STRATEGY.get("divergence", {}).get("peak_min_distance", 5)

# 成交量策略参数
VOL_RATIO_THRESHOLD = STRATEGY.get("volume", {}).get("ratio_threshold", 1.50)
VOL_RATIO_SHRINK = STRATEGY.get("volume", {}).get("ratio_shrink", 0.60)

# 综合评分阈值
SCORE_STRONG_BUY = STRATEGY.get("score", {}).get("strong_buy", 3)
SCORE_BUY = STRATEGY.get("score", {}).get("buy", 2)
SCORE_SELL = STRATEGY.get("score", {}).get("sell", -2)
SCORE_STRONG_SELL = STRATEGY.get("score", {}).get("strong_sell", -3)

# 日志由调用方统一配置，本模块只获取 logger 实例
logger = logging.getLogger("MACD_Monitor")

# ETF 行情缓存（60秒有效期）
_ETF_CACHE = {"spot_df": None, "timestamp": 0}


def _get_etf_spot_df():
    """获取 ETF 实时行情列表，带60秒缓存。"""
    now = time.time()
    if _ETF_CACHE["spot_df"] is None or now - _ETF_CACHE["timestamp"] > 60:
        _ETF_CACHE["spot_df"] = ak.fund_etf_spot_em()
        _ETF_CACHE["timestamp"] = now
    return _ETF_CACHE["spot_df"]


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
    rsv = rsv.fillna(50)  # 前 n-1 个值用中性值填充

    # K 值: 当日K = 2/3 * 前一日K + 1/3 * 当日RSV
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()

    # D 值: 当日D = 2/3 * 前一日D + 1/3 * 当日K
    d = k.ewm(alpha=1 / m2, adjust=False).mean()

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


def compute_ma_trend(df: pd.DataFrame, ma_period: int = 20) -> str:
    """
    计算 MA 均线趋势方向。
    返回: 'up'(价格在均线上方), 'down'(价格在均线下方), 'neutral'(中性区)
    """
    if len(df) < ma_period:
        return "neutral"
    ma = df["close"].rolling(window=ma_period, min_periods=ma_period).mean()
    latest_price = df["close"].iloc[-1]
    latest_ma = ma.iloc[-1]
    if latest_price > latest_ma * 1.02:
        return "up"
    elif latest_price < latest_ma * 0.98:
        return "down"
    return "neutral"


# ===================== 数据获取 =====================
def fetch_history_daily(symbol: str = STOCK_CODE, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """
    获取ETF历史日K线数据（含 open/high/low/close/volume）。
    使用 ak.fund_etf_hist_em（东财ETF数据源）。
    """
    logger.info(f"正在获取 {symbol} 历史日K线数据...")

    try:
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
            try:
                df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
                df = df.tail(days).copy()
            except Exception as e2:
                logger.error(f"fund_etf_hist_em 获取全部历史也失败: {e2}")
                return pd.DataFrame()

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
    except Exception as e:
        logger.error(f"获取历史日K线数据失败: {e}")
        return pd.DataFrame()


def _get_exchange_prefix(code: str) -> str:
    """根据 ETF 代码判断交易所前缀：sh(上海) / sz(深圳)。"""
    if code.startswith(("51", "58")):
        return "sh"
    elif code.startswith(("15", "16", "18")):
        return "sz"
    return "sh"


def fetch_history_minute(symbol: str, period: str) -> pd.DataFrame:
    """
    获取 ETF 分钟级 K 线数据（东财数据源）。
    period: 1min, 5min, 15min, 30min, 60min
    """
    logger.info(f"正在获取 {symbol} {period} 分钟K线数据...")

    try:
        prefix = _get_exchange_prefix(symbol)
        full_symbol = f"{prefix}{symbol}"
        min_period = period.replace("min", "")  # "1min" -> "1"

        # 分钟级数据历史有限，按周期取最近足够天数
        days_map = {"1min": 5, "5min": 20, "15min": 60, "30min": 120, "60min": 240}
        fetch_days = days_map.get(period, 5)

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=fetch_days)).strftime("%Y%m%d")

        try:
            df = ak.fund_etf_hist_min_em(
                symbol=full_symbol,
                period=min_period,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
        except Exception as e:
            logger.warning(f"fund_etf_hist_min_em 获取分钟数据失败: {e}")
            return pd.DataFrame()

        df = df.rename(columns={
            "时间": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"{period} 数据获取完成，共 {len(df)} 条记录，最新收盘: {df['close'].iloc[-1]:.3f}")
        return df
    except Exception as e:
        logger.error(f"获取分钟K线数据失败: {e}")
        return pd.DataFrame()


def fetch_history(symbol: str = STOCK_CODE, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """统一历史数据入口，根据 config 中的 period 自动选择日线或分钟线。"""
    period = CONFIG.get("monitor", {}).get("period", "daily")
    if period == "daily":
        return fetch_history_daily(symbol, days)
    return fetch_history_minute(symbol, period)


def fetch_spot_price(symbol: str = STOCK_CODE) -> Optional[Tuple[float, str, str]]:
    """
    获取ETF实时行情快照。
    返回: (最新价, 数据日期, 更新时间) 或 None
    """
    try:
        df = _get_etf_spot_df()
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


def fetch_etf_name(symbol: str = STOCK_CODE) -> str:
    """
    从 fund_etf_spot_em 接口获取 ETF 名称。
    返回: ETF 名称 或空字符串
    """
    try:
        df = _get_etf_spot_df()
        row = df[df["代码"] == symbol]
        if not row.empty:
            return str(row["名称"].iloc[0])
    except Exception as e:
        logger.warning(f"获取 ETF 名称失败: {e}")
    return ""


# ===================== MACD 信号检测 =====================
def detect_significant_bar(macd_df: pd.DataFrame, lookback: int = BAR_RECENT_N,
                           ratio: float = BAR_SIGNIFICANT_RATIO) -> str:
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
    if len(macd_df) < PEAK_MIN_DISTANCE * 2 + 1:
        return ""

    if len(macd_df) < lookback:
        lookback = len(macd_df) - 5

    sub_df = macd_df.iloc[-lookback:].copy()
    close = sub_df["close"]
    hist = sub_df["MACD_HIST"]
    dif = sub_df["DIF"]

    peaks, troughs = find_peaks_and_troughs(close, min_distance=PEAK_MIN_DISTANCE)

    signal = ""

    # 顶背离：遍历最近 4 个高点的所有两两组合（解决三重顶漏检）
    recent_peaks = peaks[-4:] if len(peaks) > 4 else peaks
    if len(recent_peaks) >= 2:
        for i in range(len(recent_peaks) - 1):
            for j in range(i + 1, len(recent_peaks)):
                p1, p2 = recent_peaks[i], recent_peaks[j]
                price1, price2 = close.iloc[p1], close.iloc[p2]
                hist1, hist2 = hist.iloc[p1], hist.iloc[p2]
                dif1, dif2 = dif.iloc[p1], dif.iloc[p2]
                if price2 > price1 and (hist2 < hist1 or dif2 < dif1):
                    signal = f"MACD顶背离(价:{price1:.3f}->{price2:.3f}, HIST:{hist1:.4f}->{hist2:.4f})【建议减仓/观望】"
                    break
            if signal:
                break

    # 底背离：遍历最近 4 个低点的所有两两组合
    if not signal:
        recent_troughs = troughs[-4:] if len(troughs) > 4 else troughs
        if len(recent_troughs) >= 2:
            for i in range(len(recent_troughs) - 1):
                for j in range(i + 1, len(recent_troughs)):
                    t1, t2 = recent_troughs[i], recent_troughs[j]
                    price1, price2 = close.iloc[t1], close.iloc[t2]
                    hist1, hist2 = hist.iloc[t1], hist.iloc[t2]
                    dif1, dif2 = dif.iloc[t1], dif.iloc[t2]
                    if price2 < price1 and (hist2 > hist1 or dif2 > dif1):
                        signal = f"MACD底背离(价:{price1:.3f}->{price2:.3f}, HIST:{hist1:.4f}->{hist2:.4f})【建议买进/加仓】"
                        break
                if signal:
                    break

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
    if len(price_df) < PEAK_MIN_DISTANCE * 2 + 1 or len(kdj_df) < PEAK_MIN_DISTANCE * 2 + 1:
        return ""

    if len(price_df) < lookback or len(kdj_df) < lookback:
        lookback = min(len(price_df), len(kdj_df)) - 5

    close = price_df["close"].iloc[-lookback:]
    k_values = kdj_df["K"].iloc[-lookback:]

    peaks, troughs = find_peaks_and_troughs(close, min_distance=PEAK_MIN_DISTANCE)

    # 顶背离：遍历最近 4 个高点的所有两两组合
    recent_peaks = peaks[-4:] if len(peaks) > 4 else peaks
    if len(recent_peaks) >= 2:
        for i in range(len(recent_peaks) - 1):
            for j in range(i + 1, len(recent_peaks)):
                p1, p2 = recent_peaks[i], recent_peaks[j]
                price1, price2 = close.iloc[p1], close.iloc[p2]
                k1, k2 = k_values.iloc[p1], k_values.iloc[p2]
                if price2 > price1 and k2 < k1:
                    return f"KDJ顶背离(价:{price1:.3f}->{price2:.3f}, K:{k1:.2f}->{k2:.2f})【减仓警示】"

    # 底背离：遍历最近 4 个低点的所有两两组合
    recent_troughs = troughs[-4:] if len(troughs) > 4 else troughs
    if len(recent_troughs) >= 2:
        for i in range(len(recent_troughs) - 1):
            for j in range(i + 1, len(recent_troughs)):
                t1, t2 = recent_troughs[i], recent_troughs[j]
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
    vol_ratio = vol_info["VOL_RATIO"].iloc[-1] if not vol_info.empty and not pd.isna(vol_info["VOL_RATIO"].iloc[-1]) else 1.0

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
        vol_signal: str,
        price_df: pd.DataFrame,
) -> Tuple[int, str, str]:
    """
    多指标综合评分系统。

    评分规则:
        买入加分项:
            +1: MACD金叉 / 显著红柱
            +1: MACD底背离
            +1: KDJ低位金叉(<30) / KDJ底背离 / J<0超卖
            +1: 放量上涨(量比>1.2且涨幅>1%)
            +1: 缩量止跌/缩量盘整(量比<0.5)

        卖出减分项:
            -1: MACD死叉 / 显著绿柱
            -1: MACD顶背离
            -1: KDJ高位死叉(>70) / KDJ顶背离 / J>100超买
            -1: 放量下跌(量比>1.2且跌幅>1%)
            -1: 缩量上涨(量比<0.5且价格上涨)

    理论最大加分/减分: +/-7
    返回: (score, score_desc, reasons)
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

    # --- KDJ 评分（带 MA20 趋势过滤） ---
    if len(kdj_df) >= 2:
        latest_kdj = kdj_df.iloc[-1]
        prev_kdj = kdj_df.iloc[-2]
        k, d, j = latest_kdj["K"], latest_kdj["D"], latest_kdj["J"]
        prev_k = prev_kdj["K"]

        # 趋势过滤：上涨市中忽略 KDJ 死叉，下跌市中忽略 KDJ 金叉
        trend = compute_ma_trend(price_df) if price_df is not None else "neutral"

        # 金叉/死叉（已按报告降低 KDJ 权重，避免单一指标过度驱动）
        if prev_k < prev_kdj["D"] and k > d:
            if trend == "down":
                reasons.append("KDJ金叉(被趋势过滤，下跌市中忽略)")
            else:
                if k < 30:
                    score += 1
                    reasons.append("KDJ低位金叉(+1)")
                else:
                    score += 1
                    reasons.append("KDJ金叉(+1)")
        elif prev_k > prev_kdj["D"] and k < d:
            if trend == "up":
                reasons.append("KDJ死叉(被趋势过滤，上涨市中忽略)")
            else:
                if k > 70:
                    score -= 1
                    reasons.append("KDJ高位死叉(-1)")
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
    elif "缩量下跌" in vol_signal or "缩量盘整" in vol_signal:
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


def _position_suggestion(score: int) -> str:
    """根据评分映射建议仓位比例。"""
    if score >= SCORE_STRONG_BUY:
        return "建议仓位: 80%~100%"
    elif score >= SCORE_BUY:
        return "建议仓位: 50%~80%"
    elif score <= SCORE_STRONG_SELL:
        return "建议仓位: 0% (清仓)"
    elif score <= SCORE_SELL:
        return "建议仓位: 0%~20%"
    else:
        return "建议仓位: 20%~50%"


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

    advice += f" | {_position_suggestion(score)}"

    if reasons:
        advice += f" 依据: {', '.join(reasons)}"

    return advice


# ===================== 主监控循环 =====================
def main():
    """
    主入口已移至 main.py，本模块作为纯库使用。
    请运行: python main.py -c <ETF代码>
    """
    logger.info("本模块为纯库文件，请通过 main.py 启动监控。")
    logger.info("用法: python main.py -c 518880")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("监控程序被用户中断，退出。")
