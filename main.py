"""ETF 多指标实时监控 - 命令行入口

包装 macd_monitor_518880 模块，支持通过命令行参数指定标的、轮询间隔等。
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import akshare as ak

from macd_monitor_518880 import (
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    KDJ_N, KDJ_M1, KDJ_M2, VOL_MA_DAYS,
    fetch_history_daily, fetch_spot_price,
    compute_macd, compute_kdj, compute_volume_signals,
    detect_significant_bar, detect_divergence,
    detect_kdj_signals, detect_volume_signals,
    calculate_comprehensive_score, generate_comprehensive_advice,
)

LOG_DIR = Path("logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETF 多指标实时监控（MACD + KDJ + 成交量）",
    )
    parser.add_argument("-c", "--code", default="518880",
                        help="ETF 代码，默认 518880（华安黄金ETF）")
    parser.add_argument("-n", "--name", default="",
                        help="ETF 名称（仅用于日志显示）")
    parser.add_argument("-i", "--interval", type=int, default=60,
                        help="轮询间隔（秒），默认 60")
    parser.add_argument("-d", "--history-days", type=int, default=120,
                        help="历史 K 线获取天数，默认 120")
    parser.add_argument("--once", action="store_true",
                        help="只执行一次检测，不进入循环")
    return parser.parse_args()


def setup_logger(code: str) -> logging.Logger:
    """配置日志：统一输出到 logs/ 目录，并覆盖默认 root logger。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"monitor_{code}.log"

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


def run_check(symbol: str, name: str, history_days: int, logger: logging.Logger) -> None:
    """执行一次完整的指标检测，并以清晰排版输出结果。"""
    hist_df = fetch_history_daily(symbol, history_days)

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
        vol_signal,
    )
    advice = generate_comprehensive_advice(score, score_desc, reasons)

    # 最新值
    latest_macd = macd_df.iloc[-1]
    latest_kdj = kdj_df.iloc[-1]
    latest_vol = vol_info.iloc[-1]
    latest_hist = hist_df.iloc[-1]

    label = f"{symbol}({name})" if name else symbol
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    ]

    _log_block(logger, lines)


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.code)
    label = args.code + (f"({args.name})" if args.name else "")

    logger.info("=" * 60)
    logger.info(f"ETF 多指标监控启动 | 标的: {label}")
    logger.info(f"AKShare 版本: {ak.__version__}")
    logger.info(
        f"轮询间隔: {args.interval}s | 历史天数: {args.history_days} | "
        f"模式: {'单次' if args.once else '循环'}"
    )
    logger.info("=" * 60)

    if args.once:
        try:
            run_check(args.code, args.name, args.history_days, logger)
        except Exception as e:
            logger.error(f"检测失败: {e}", exc_info=True)
            return 1
        return 0

    loop_count = 0
    while True:
        loop_count += 1
        try:
            logger.info(f"--- 第 {loop_count} 次检测 ---")
            run_check(args.code, args.name, args.history_days, logger)
        except Exception as e:
            logger.error(f"主循环异常: {e}", exc_info=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n监控已中断，退出。")
        sys.exit(0)
