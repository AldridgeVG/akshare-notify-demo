"""ETF 多指标实时监控 - 命令行入口

包装 macd_monitor_518880 模块，支持通过命令行参数指定标的、轮询间隔等。
"""
import argparse
import logging
import sys
import time

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"monitor_{code}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("ETF_Monitor")


def run_check(symbol: str, name: str, history_days: int, logger: logging.Logger) -> None:
    """执行一次完整的指标检测并输出建议。"""
    hist_df = fetch_history_daily(symbol, history_days)

    spot = fetch_spot_price(symbol)
    if spot is not None:
        latest_price, data_date_str, data_time_str = spot
        logger.info(f"实时价格: {latest_price:.3f} | 时间: {data_date_str} {data_time_str}")
        hist_df = hist_df.copy()
        hist_df.loc[hist_df.index[-1], "close"] = latest_price

    macd_df = compute_macd(hist_df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    kdj_df = compute_kdj(hist_df, KDJ_N, KDJ_M1, KDJ_M2)
    vol_info = compute_volume_signals(hist_df, VOL_MA_DAYS)

    bar_signal = detect_significant_bar(macd_df)
    div_signal = detect_divergence(macd_df)
    kdj_signal, kdj_div = detect_kdj_signals(kdj_df, hist_df)
    vol_signal = detect_volume_signals(hist_df, vol_info)

    score, score_desc, reasons = calculate_comprehensive_score(
        macd_df, bar_signal, div_signal,
        kdj_df, kdj_signal, kdj_div,
        vol_signal,
    )
    advice = generate_comprehensive_advice(score, score_desc, reasons)

    label = f"{symbol}({name})" if name else symbol
    latest_macd = macd_df.iloc[-1]
    latest_kdj = kdj_df.iloc[-1]
    latest_vol = vol_info.iloc[-1]

    logger.info(
        f"[{label}] MACD | DIF={latest_macd['DIF']:.4f} "
        f"DEA={latest_macd['DEA']:.4f} HIST={latest_macd['MACD_HIST']:.4f}"
    )
    logger.info(
        f"[{label}] KDJ  | K={latest_kdj['K']:.2f} "
        f"D={latest_kdj['D']:.2f} J={latest_kdj['J']:.2f}"
    )
    logger.info(f"[{label}] VOL  | 量比={latest_vol['VOL_RATIO']:.2f}")

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
