# ChatCaht Dashboard

ChatCaht 全链路监控看板 —— 独立只读子项目，实时展示 WakeUp / SpText / LoLLama / LM Studio / GVoice / ChatCaht 编排器的运行状态、性能指标与日志。

## 特点

- **零侵入**：不 import `chatcaht` 包、不修改主项目任何文件；只读取 `configs/config.yaml`、`artifacts/run/*.pid.json` 与各项目 `artifacts/logs/*.log`
- **实时状态**：SpText / LoLLama 走 `status` 命令拿富状态（ready、listening、backend、模型、5 层记忆统计、uptime、已服务请求数），WakeUp 解析连接欢迎帧（自带 listening/model）+ ping，GVoice 只发 ping（它不支持 status，乱发会在其日志刷 WARNING），LM Studio 走 HTTP `/v1/models`；每 3s 刷新
- **管线视图**：唤醒 → 识别 → 编排 → 智能体 → 模型 → 合成 的链路动画，一眼看出断在哪一环
- **性能指标**：自动从 `chatcaht.log` 解析 LLM 首字延迟、总耗时、LoLLama 回复耗时、TTS 合成耗时，绘制走势图
- **日志查错**：分服务 tab、级别过滤（可只看错误）、全文搜索、traceback 聚合展示、自动跟随；卡片上的"定位错误"一键跳到对应日志的错误行
- **双日志源**：同时接入 ChatCaht 托管日志（`*.service.log`，supervisor 拉起时的 stdout）和各项目自有日志（tab 上带 ⌂ 标记，服务独立运行时以这份为准）；卡片摘要自动选用更新鲜的那份
- **统一格式 + 兼容历史**：全家日志已统一为 `2026-07-06 10:09:29,554 INFO logger: 消息`（UTF-8）；解析器同时兼容旧格式（SpText/GVoice 的 `LEVEL [logger]`、WakeUp 的 `HH:MM:SS [LEVEL]`），历史日志照常可查
- **用户记忆展示**：读取 LoLLama `artifacts/memory/{user_id}.json`，按 4 层（情景/事实/偏好/画像）展示记忆条目，实时计算有效强度（按 LoLLama 同款半衰期公式），支持层过滤与多用户切换
- **响应式**：桌面多列网格，窄屏自动折叠为单列

## 启动

在 ChatCaht 根目录：

```powershell
uv run python dashboard/server.py
# 打开 http://127.0.0.1:8899
```

可选参数：`--host 0.0.0.0`（局域网访问）、`--port 9000`。

依赖仅为主项目 venv 已有的 `pyyaml` + `websockets`，其余全部标准库。

## 接口

| 路径 | 说明 |
| --- | --- |
| `GET /api/status` | 各项目进程/健康/日志摘要（带 2.5s 缓存，多端打开不会打爆底层服务） |
| `GET /api/metrics` | 从 chatcaht.log 尾部解析的延迟序列 |
| `GET /api/logs?name=X.log&lines=400&level=ERROR&q=关键词` | 日志尾部查询（项目自有日志用 `项目:文件.log` 形式，如 `sptext:sptext.log`） |
| `GET /api/logfiles` | 可用日志文件列表（含托管与项目自有两类） |
| `GET /api/memory` | LoLLama 用户记忆条目（4 层 + 实时有效强度） |

## 状态语义

- **运行中**：健康检查通过且近期日志无错误
- **异常**：健康检查通过但近期日志有 ERROR，或进程存活但健康检查不通过
- **离线**：健康检查失败且进程不在
- **待机**：未启用（如 provider 非 lollama 时的 LoLLama）或仅按日志活动判断的编排器处于空闲
