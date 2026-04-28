# akshare-notify-demo

基于 [AKShare](https://github.com/akfamily/akshare) 的 ETF 多指标实时监控工具，结合 **MACD + KDJ + 成交量** 给出综合操作建议。

默认监控标的：华安黄金 ETF（**518880**）。

## 安装

```bash
pip install akshare pandas numpy
```

## 使用

监控默认标的（518880）并循环轮询：

```bash
python main.py
```

切换其他 ETF（示例：沪深 300 ETF 510300）：

```bash
python main.py -c 510300 -n 沪深300ETF
```

只执行一次检测：

```bash
python main.py --once
```

### 命令行参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-c, --code` | `518880` | ETF 代码 |
| `-n, --name` | 空 | ETF 名称（仅日志显示） |
| `-i, --interval` | `60` | 轮询间隔（秒） |
| `-d, --history-days` | `120` | 历史 K 线天数 |
| `--once` | 关 | 只执行一次后退出 |

## 主要功能

- **MACD**：金叉/死叉、显著红绿柱、顶/底背离
- **KDJ**：金叉/死叉、超买/超卖、背离
- **成交量**：放量/缩量、量价配合判断
- **综合评分**：多指标加权打分，输出"强烈看多 / 偏多 / 观望 / 偏空 / 强烈看空"建议

## 输出

日志同时输出到控制台和 `monitor_<code>.log` 文件。

## 项目结构

```
akshare-notify-demo/
├── main.py                    # 命令行入口
├── macd_monito.py     # 指标计算与信号检测核心模块
└── README.md
```

## 免责声明

本项目仅供学习和技术研究使用，**不构成任何投资建议**。投资有风险，决策需谨慎。
