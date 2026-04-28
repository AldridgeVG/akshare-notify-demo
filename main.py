"""ETF 多指标实时监控 - 命令行入口

支持单标的或多标的串行轮询，名称自动从 AKShare 接口获取。
"""
import argparse
import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import akshare as ak

from config_loader import CONFIG
from macd_monitor import (
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    KDJ_N, KDJ_M1, KDJ_M2, VOL_MA_DAYS,
    SCORE_STRONG_BUY, SCORE_BUY, SCORE_SELL, SCORE_STRONG_SELL,
    fetch_history, fetch_spot_price, fetch_etf_name,
    compute_macd, compute_kdj, compute_volume_signals,
    detect_significant_bar, detect_divergence,
    detect_kdj_signals, detect_volume_signals,
    calculate_comprehensive_score, generate_comprehensive_advice,
)

LOG_DIR = Path("logs")

# 从 config.yml 读取默认值，命令行参数可覆盖
_MONITOR_CFG = CONFIG.get("monitor", {})

# 信号确认/防抖状态（code -> deque of recent score levels）
_SCORE_HISTORY: dict[str, deque[str]] = {}
# 已确认的评分级别（code -> level）
_CONFIRMED_LEVEL: dict[str, str] = {}


def _score_level(score: int) -> str:
    """将评分映射为离散级别，用于防抖确认。"""
    if score >= SCORE_STRONG_BUY:
        return "strong_buy"
    elif score >= SCORE_BUY:
        return "buy"
    elif score <= SCORE_STRONG_SELL:
        return "strong_sell"
    elif score <= SCORE_SELL:
        return "sell"
    else:
        return "neutral"


def _apply_signal_confirmation(code: str, score: int, score_desc: str, advice: str) -> tuple[str, str]:
    """
    信号确认防抖：评分级别需连续 2 次一致才确认切换。
    未确认时，在描述与建议中追加待确认标识。
    """
    level = _score_level(score)
    history = _SCORE_HISTORY.setdefault(code, deque(maxlen=2))
    history.append(level)

    confirmed_level = _CONFIRMED_LEVEL.get(code, level)

    # 连续 2 次同一级别则更新确认状态
    if len(history) == 2 and history[0] == history[1]:
        confirmed_level = level
        _CONFIRMED_LEVEL[code] = confirmed_level

    if level == confirmed_level:
        return score_desc, advice

    # 级别切换尚未确认
    pending_desc = score_desc + " (待确认)"
    pending_advice = advice + " [信号切换中，建议观望]"
    return pending_desc, pending_advice


def _default_interval() -> int:
    """根据配置的 period 自动选择默认轮询间隔。"""
    period = _MONITOR_CFG.get("period", "daily")
    return {
        "daily": 60,
        "1min": 30,
        "5min": 120,
        "15min": 300,
        "30min": 600,
        "60min": 1200,
    }.get(period, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETF 多指标实时监控（MACD + KDJ + 成交量）",
    )
    parser.add_argument("-c", "--code", default="",
                        help="单个 ETF 代码，例如 518880")
    parser.add_argument("-cs", "--codes", default="",
                        help="多个 ETF 代码，逗号分隔，例如 518880,512880")
    parser.add_argument("-i", "--interval", type=int,
                        default=_MONITOR_CFG.get("poll_interval") or _default_interval(),
                        help="轮询间隔（秒），默认根据周期自动选择")
    parser.add_argument("-d", "--history-days", type=int,
                        default=_MONITOR_CFG.get("history_days", 120),
                        help="历史 K 线获取天数，默认 120")
    parser.add_argument("--once", action="store_true",
                        help="只执行一次检测，不进入循环")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    """配置日志：统一输出到 logs/monitor.log，并覆盖默认 root logger。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "monitor.log"

    root = logging.getLogger()
    # 清除已有 handlers，避免 macd_monitor_518880 模块的 basicConfig 干扰
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    return logging.getLogger("ETF_Monitor")


def _log_block(logger: logging.Logger, lines: list[str]) -> None:
    """逐行记录，确保每行都有时间戳前缀。"""
    for line in lines:
        logger.info(line)


def run_check(symbol: str, history_days: int, logger: logging.Logger) -> None:
    """执行一次完整的指标检测，并以清晰排版输出结果。"""
    # 获取名称（首次从接口读取）
    name = fetch_etf_name(symbol)
    hist_df = fetch_history(symbol, history_days)

    if hist_df is None or hist_df.empty:
        logger.warning(f"[{symbol}] 未能获取历史数据，跳过本次检测")
        return

    # 实时价格
    spot = fetch_spot_price(symbol)
    if spot is not None:
        latest_price, data_date_str, data_time_str = spot
        price_info = (
            f"实时价格: {latest_price:.3f}  |  "
            f"数据时间: {data_date_str} {data_time_str}"
        )
        hist_df = hist_df.copy()
        hist_df.loc[hist_df.index[-1], "close"] = latest_price
    else:
        latest_price = hist_df["close"].iloc[-1]
        price_info = f"实时价格: 未获取  |  最新收盘: {latest_price:.3f}"

    # 计算指标
    macd_df = compute_macd(hist_df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    kdj_df = compute_kdj(hist_df, KDJ_N, KDJ_M1, KDJ_M2)
    vol_info = compute_volume_signals(hist_df, VOL_MA_DAYS)

    # 检测信号
    bar_signal = detect_significant_bar(macd_df)
    div_signal = detect_divergence(macd_df)
    kdj_signal, kdj_div = detect_kdj_signals(kdj_df, hist_df)
    vol_signal = detect_volume_signals(hist_df, vol_info)

    # 综合评分
    score, score_desc, reasons = calculate_comprehensive_score(
        macd_df, bar_signal, div_signal,
        kdj_df, kdj_signal, kdj_div,
        vol_signal, hist_df,
    )
    advice = generate_comprehensive_advice(score, score_desc, reasons)

    # 信号确认防抖（B2）
    score_desc, advice = _apply_signal_confirmation(symbol, score, score_desc, advice)

    # 最新值
    latest_macd = macd_df.iloc[-1]
    latest_kdj = kdj_df.iloc[-1]
    latest_vol = vol_info.iloc[-1]
    latest_hist = hist_df.iloc[-1]

    label = f"{symbol}({name})" if name else symbol
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 收集强烈信号
    strong_signals = []
    if bar_signal and "显著" in bar_signal:
        strong_signals.append(f"[MACD柱体] {bar_signal}")
    if div_signal:
        strong_signals.append(f"[MACD背离] {div_signal}")
    if kdj_signal and ("强买入" in kdj_signal or "强卖出" in kdj_signal):
        strong_signals.append(f"[KDJ信号]  {kdj_signal}")
    if kdj_div:
        strong_signals.append(f"[KDJ背离]  {kdj_div}")
    if vol_signal and ("放量上涨" in vol_signal or "放量下跌" in vol_signal):
        strong_signals.append(f"[成交信号] {vol_signal}")
    if score >= SCORE_STRONG_BUY:
        strong_signals.append(f"[综合评分] {score_desc}")
    elif score <= SCORE_STRONG_SELL:
        strong_signals.append(f"[综合评分] {score_desc}")

    # 构建块式日志
    lines = [
        "=" * 60,
        f"  标的: {label}  |  检测时间: {now_str}",
        "-" * 60,
        f"  {price_info}",
        "-" * 60,
        "  [技术指标]",
        f"    MACD   DIF: {latest_macd['DIF']:>8.4f}   DEA: {latest_macd['DEA']:>8.4f}   HIST: {latest_macd['MACD_HIST']:>8.4f}",
        f"    KDJ      K: {latest_kdj['K']:>8.2f}      D: {latest_kdj['D']:>8.2f}      J: {latest_kdj['J']:>8.2f}",
        f"    成交   量比: {latest_vol['VOL_RATIO']:>8.2f}   均量: {latest_vol['VOL_MA']:>10,.0f}   现量: {latest_hist['volume']:>10,.0f}",
        "-" * 60,
    ]

    if strong_signals:
        lines.append("  ⚠ 强烈信号")
        for sig in strong_signals:
            lines.append(f"    >>> {sig}")
        lines.append("-" * 60)

    lines.extend([
        "  [信号检测]",
        f"    [MACD柱体] {bar_signal if bar_signal else '(无)'}",
        f"    [MACD背离] {div_signal if div_signal else '(无)'}",
        f"    [KDJ信号]  {kdj_signal if (kdj_signal and '中性' not in kdj_signal) else '(无)'}",
        f"    [KDJ背离]  {kdj_div if kdj_div else '(无)'}",
        f"    [成交信号] {vol_signal if vol_signal else '(无)'}",
        "-" * 60,
        f"  [综合评分] {score_desc}",
        f"  [评分依据] {', '.join(reasons) if reasons else '无'}",
        "-" * 60,
        f"  [操作建议] {advice}",
        "=" * 60,
    ])

    _log_block(logger, lines)


def main() -> int:
    args = parse_args()

    # 解析代码列表：-cs 优先，否则回退到 -c，最后再回退到 config.yml
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    elif args.code:
        codes = [args.code.strip()]
    else:
        codes = [CONFIG.get("stock", {}).get("code", "518880")]

    logger = setup_logger()

    logger.info("=" * 60)
    logger.info(f"ETF 多指标监控启动 | 标的数: {len(codes)} | 列表: {', '.join(codes)}")
    logger.info(f"AKShare 版本: {ak.__version__}")
    logger.info(
        f"轮询间隔: {args.interval}s | 历史天数: {args.history_days} | "
        f"模式: {'单次' if args.once else '循环'}"
    )
    logger.info("=" * 60)

    if args.once:
        for code in codes:
            try:
                run_check(code, args.history_days, logger)
            except Exception as e:
                logger.error(f"[{code}] 检测失败: {e}", exc_info=True)
        return 0

    loop_count = 0
    while True:
        loop_count += 1
        logger.info(f"--- 第 {loop_count} 轮检测 ---")
        for code in codes:
            try:
                run_check(code, args.history_days, logger)
            except Exception as e:
                logger.error(f"[{code}] 检测异常: {e}", exc_info=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n监控已中断，退出。")
        sys.exit(0)
