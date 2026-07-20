# ChatCaht

## 统一音频运行时

Windows x64 语音模式默认使用 `audio.mode: unified_required`。`AudioRuntime` 是唯一打开默认通信麦克风和扬声器的进程；它用 WASAPI 持有设备，以 WebRTC APM AEC3 处理实际播放参考信号，并把 16 kHz/单声道/10 ms 的清洁 PCM 同时路由给 WakeUp 与 SpText。

状态机仍是 `唤醒 -> STT -> LLM/TTS -> 可打断回复 -> 返回唤醒`。播放期间只有 AudioRuntime 连续确认 120 ms 近端人声后发出的 `near_end_start(speech_id)` 才能触发一次打断；ASR partial 不再负责打断。200 ms 前导音频会在确认后回灌，播放或取消后保留 200 ms AEC tail guard。

一次唤醒应答或助手回复只建立一个连续播放流：多个 GVoice PCM 块在同一个 `play_start` / `play_end` 生命周期内进入有界队列，`play_cancel` 通过播放代际令牌使已经阻塞的旧块立即失效，避免打断后复播。

AEC 和设备初始化是硬依赖。`chatcaht run` 会先启动 AudioRuntime，再启动 WakeUp、SpText 和 GVoice；`ready` 或 `aec_ready` 为 false 时不会进入旧的全双工链路。`chatcaht doctor` 会把该故障显示为 `audio(aec)`。`--no-audio`、文本命令和 mock 测试不会启动侧车；`--no-audio` 使用 WakeUp/SpText 的独立麦克风模式。

`configs/config.yaml` 的 `audio:` 段是音频参数的唯一来源。托管启动会原子生成 `artifacts/run/audio-runtime.generated.json`，AudioRuntime 在握手状态中回报实际生效参数；任何配置漂移或旧协议版本都会使健康检查失败。旧配置中的 `services.audio_runtime_config` 会被兼容忽略。

AudioRuntime 会监测默认通信设备变化、设备失活、连续 WASAPI 错误和采集停帧；异常时主动退出，由 Supervisor 完整重建原生音频进程。`audio.capture_stall_timeout_ms` 默认 3000 ms。若配置了明确的 `input_device` / `output_device`，系统默认设备变化不会影响该绑定，但设备失活与停帧检测仍然有效。

先构建原生侧车：

```powershell
cd ..\AudioRuntime
uvx --from cmake cmake -S . -B build -A x64
uvx --from cmake cmake --build build --config Release --target chat-audio-runtime --parallel 1
```

托管模式要求 WakeUp 配置为 `external_pcm`，SpText 以 `--no-listen` 启动。切换唤醒/对话只改变 PCM 路由，不会重新打开声卡。

ChatCaht 是一个本地全双工语音对话编排器，负责把唤醒词检测、语音识别、文本生成和语音合成串成一条完整链路，形成可实时打断的语音聊天体验。

它默认对接这些组件：

- `WakeUp / Wakord`：唤醒词检测
- `SpText`：实时 ASR 语音识别
- `GVoice`：流式中文 TTS 语音合成
- OpenAI-compatible API：可对接 OpenAI、LM Studio 或其他兼容 `/v1/chat/completions` 的模型服务
- `LoLLama`（可选，`llm.provider: lollama`）：全双工 LLM 智能体服务——5 层记忆系统（工作/情景/语义/程序性/核心画像，带热度晋升与遗忘）+ 本地工具调用（时间/计算器/文件/记忆/shell），由它再请求 LM Studio

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
- `llm.provider`：回复生成后端，`openai`（直连 LM Studio）或 `lollama`（经 LoLLama 智能体：分层记忆 + 工具调用）
- `lollama.url` / `lollama.response_timeout_sec`：LoLLama WebSocket 地址与单条回复超时；记忆与工具参数在 `../LoLLama/configs/config.yaml` 中调
- `lollama.announce_status`：把 LoLLama 的状态钩子（"我算一下""让我想想"等执行进度）送 TTS 播报；各阶段文案在 LoLLama 侧 `status.announce` 段配置
- `openai.extra_body`：附加请求字段，原样并入 `/chat/completions` 请求体；思考型模型（如 Ollama 上的 qwen3.5）需要 `reasoning_effort: none` 关闭思考，否则回复内容为空且首字延迟极高
- `wake.url`：WakeUp WebSocket 地址
- `stt.url`：SpText WebSocket 地址
- `stt.final_events_only`：默认只把 SpText 的 final 转写交给模型，避免用户话没说完就开始回复
- `stt.partial_fallback_sec`：SpText 长期不产生 final 时的兜底停顿时间；默认 `1.2` 秒，太小会把一句话切成多轮，设为 `0` 可禁用兜底提交
- `tts.url`：GVoice WebSocket 地址
- `duplex.allow_barge_in`：是否允许打断
- `duplex.end_session_words`：结束会话口令
- `duplex.loop_forever`：会话结束后回到待唤醒状态（随时可再次唤醒）
- `duplex.idle_timeout_sec`：对话中连续无语音超过该秒数自动回到待唤醒，0 禁用
- `duplex.wake_ack_text`：唤醒成功后播报的应答语（如"我在"），留空不播报
- `duplex.max_history_turns` / `duplex.reset_history_per_session`：ChatCaht 直连 `openai` 时维护的短期对话窗口；`llm.provider: lollama` 时 ChatCaht 只发送当前文本，用户身份、对话窗口、长期记忆和工具调用都由 LoLLama 侧维护，ChatCaht 只保留一小段 ASR 轮次缓存用于剥离 SpText 累计前缀
- `runtime.reconnect_initial_delay_sec` / `reconnect_max_delay_sec`：断线重连退避
- `supervisor.*`：守护模式的服务托管、健康检查间隔与崩溃重启退避
- `services.*_dir`：三个底层项目的路径

## 使用

### 推荐：长期运行守护模式

一条命令拉起全部底层服务并进入"随时唤醒"待命状态，服务或会话异常均自动恢复：

```powershell
uv run chatcaht run
```

服务统一使用 `STARTING / READY / DEGRADED / FAILED` 生命周期。AudioRuntime、WakeUp、SpText、
GVoice/CosyVoice 和 LoLLama 全部达到 `READY` 后才创建语音会话；模型加载或 warmup 期间保持
`STARTING`，Supervisor 不会把它当作崩溃循环重启。`DEGRADED` 服务会阻止新会话，但保留进程供其自行恢复，
只有 `FAILED` 才会触发托管重启。

- 说唤醒词（如"小元"）开始对话，单独说"退出/再见/拜拜/闭嘴"或静默超时后回到待命，可反复唤醒
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

执行完整的本机启动前检查（模型哈希、CUDA/FP16、CosyVoice 导入依赖、LM Studio 模型、
声卡/AEC 和端口占用）：

```powershell
uv run chatcaht doctor --deep
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

## 监控看板

`dashboard/` 是一个独立只读子项目：网页端实时展示各服务健康状态、管线链路、LLM/TTS 延迟走势、LoLLama 用户记忆（4 层条目 + 有效强度）和分服务日志查错，不影响主项目运行：

```powershell
uv run python dashboard/server.py
# 打开 http://127.0.0.1:8899
```

详见 [dashboard/README.md](dashboard/README.md)。

## 日志

全家（ChatCaht / WakeUp / SpText / GVoice / LoLLama）已统一日志规范：

- **格式**：`2026-07-06 10:09:29,554 INFO chatcaht.orchestrator: 消息`（带毫秒；Dashboard 可直接解析做级别过滤与错误统计）
- **编码**：文件日志一律 UTF-8 滚动日志（10MB × 5 份）；各服务 stdout/stderr 均切到 UTF-8，Windows GBK 控制台下中文不乱码、不抛 UnicodeEncodeError
- 编排器日志：`artifacts/logs/chatcaht.log`，包含服务启停、WebSocket 连接、每轮 LLM 首字/总延迟、TTS 合成耗时等
- 托管服务的 stdout 捕获：`artifacts/logs/wakeup.service.log`、`sptext.service.log`、`gvoice.service.log`、`lollama.service.log`
- 各服务独立运行时写各自项目的 `artifacts/logs/*.log`（WakeUp 为 serve 模式新增的 `wakeup.log`），Dashboard 两边都接入并自动选用更新鲜的一份
- 需要更细的连接/合成细节时把 `runtime.log_level` 调成 `DEBUG`

## 说明

ChatCaht 本身不重复实现唤醒词、ASR 或 TTS 的底层模型，而是通过它们暴露的协议做编排。这样每个子项目都可以独立训练、升级和调试，而 ChatCaht 负责整体会话状态、服务管理、OpenAI-compatible 模型 API 对接和全链路 CLI。

## 许可证

MIT
