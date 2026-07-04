# ChatCaht

ChatCaht 是一个本地全双工语音对话编排器，负责把唤醒词检测、语音识别、文本生成和语音合成串成一条完整链路，形成可实时打断的语音聊天体验。

它默认对接这些组件：

- `WakeUp / Wakord`：唤醒词检测
- `SpText`：实时 ASR 语音识别
- `GVoice`：流式中文 TTS 语音合成
- OpenAI-compatible API：可对接 OpenAI、LM Studio 或其他兼容 `/v1/chat/completions` 的模型服务

## 特性

- **随时唤醒**：待唤醒 ⇄ 对话 持久状态机；说结束口令或空闲超时后自动回到待唤醒状态，再次喊唤醒词即可继续
- **全双工**：回复播报期间持续收音，支持语音打断（barge-in），打断时立即静音
- **长期可靠运行**：`chatcaht run` 守护模式——自动拉起底层服务、周期健康检查并重启不健康服务、会话崩溃指数退避自动恢复
- 唤醒/识别 WebSocket 断线自动重连（指数退避），LLM 请求失败自动重试，单句 TTS 失败跳过不中断整轮回复
- 支持一键启动、停止、查看底层服务；命令行文本对话与自检
- 适合本地离线或半离线部署

## 会话状态机

```text
        唤醒词                       结束口令 / 空闲超时
待唤醒 ────────▶ 对话（全双工，可打断） ────────────────▶ 回到待唤醒
   ▲                                                        │
   └────────────────────────────────────────────────────────┘
                     loop_forever: true 时无限循环
```

## 项目结构

```text
ChatCaht/
  configs/config.example.yaml
  src/chatcaht/
    adapters/          # wake / stt / tts 协议适配
    orchestrator.py    # 会话状态机：唤醒、对话、打断、空闲超时、重连
    supervisor.py      # 守护：服务托管、健康检查、崩溃自愈
    openai_client.py   # OpenAI-compatible 流式模型客户端
    service_manager.py # WakeUp/SpText/GVoice 进程管理
    cli.py
  tests/
```

## 安装

```powershell
cd D:\Project\Python_Project\ChatCaht
uv sync --extra dev
```

复制一份可修改配置：

```powershell
uv run chatcaht init-config --out configs/config.yaml
```

## 配置

常用配置项在 `configs/config.example.yaml` 中：

- `openai.base_url`：OpenAI-compatible API 地址，例如 `https://api.openai.com/v1` 或本地兼容服务
- `openai.api_key`：API Key；本地兼容服务可按需填写占位值
- `openai.model`：模型名
- `wake.url`：WakeUp WebSocket 地址
- `stt.url`：SpText WebSocket 地址
- `tts.url`：GVoice WebSocket 地址
- `duplex.allow_barge_in`：是否允许打断
- `duplex.end_session_words`：结束会话口令
- `duplex.loop_forever`：会话结束后回到待唤醒状态（随时可再次唤醒）
- `duplex.idle_timeout_sec`：对话中连续无语音超过该秒数自动回到待唤醒，0 禁用
- `duplex.wake_ack_text`：唤醒成功后播报的应答语（如"我在"），留空不播报
- `duplex.reset_history_per_session`：重新唤醒时是否清空对话历史
- `runtime.reconnect_initial_delay_sec` / `reconnect_max_delay_sec`：断线重连退避
- `supervisor.*`：守护模式的服务托管、健康检查间隔与崩溃重启退避
- `services.*_dir`：三个底层项目的路径

## 使用

### 推荐：长期运行守护模式

一条命令拉起全部底层服务并进入"随时唤醒"待命状态，服务或会话异常均自动恢复：

```powershell
uv run chatcaht run
```

- 说唤醒词（如"小元"）开始对话，说"退出/再见"或静默超时后回到待命，可反复唤醒
- 加 `--no-services` 表示底层服务由你自己管理，ChatCaht 只做编排
- 按 Ctrl+C 停止

### 手动管理

启动底层服务：

```powershell
uv run chatcaht services start
uv run chatcaht services status
```

停止底层服务：

```powershell
uv run chatcaht services stop
```

检查服务连通性：

```powershell
uv run chatcaht doctor
```

运行自检：

```powershell
uv run chatcaht selftest
uv run --extra dev pytest -q
```

文本对话测试：

```powershell
uv run chatcaht text 你好，简单介绍一下你自己
```

进入语音对话（前台单实例，不托管底层服务）：

```powershell
uv run chatcaht chat
```

无声模式 / 只聊一轮就退出：

```powershell
uv run chatcaht chat --no-audio
uv run chatcaht chat --once
```

## 说明

ChatCaht 本身不重复实现唤醒词、ASR 或 TTS 的底层模型，而是通过它们暴露的协议做编排。这样每个子项目都可以独立训练、升级和调试，而 ChatCaht 负责整体会话状态、服务管理、OpenAI-compatible 模型 API 对接和全链路 CLI。

## 许可证

MIT
