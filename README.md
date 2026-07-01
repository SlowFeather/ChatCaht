# ChatCaht

ChatCaht 是一个本地全双工语音对话编排器，负责把唤醒词检测、语音识别、文本生成和语音合成串成一条完整链路，形成可实时打断的语音聊天体验。

它默认对接这些组件：

- `WakeUp / Wakord`：唤醒词检测
- `SpText`：实时 ASR 语音识别
- `GVoice`：流式中文 TTS 语音合成
- `LM Studio`：OpenAI 兼容的本地模型服务

## 特性

- 唤醒后进入实时对话
- 支持语音打断和流式回复
- 支持一键启动、停止、查看底层服务
- 支持命令行文本对话与自检
- 适合本地离线或半离线部署

## 项目结构

```text
ChatCaht/
  configs/config.example.yaml
  src/chatcaht/
    adapters/
    orchestrator.py
    lmstudio.py
    service_manager.py
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

- `lmstudio.base_url`：LM Studio OpenAI-compatible 地址
- `lmstudio.model`：当前加载的模型名
- `wake.url`：WakeUp WebSocket 地址
- `stt.url`：SpText WebSocket 地址
- `tts.url`：GVoice WebSocket 地址
- `duplex.allow_barge_in`：是否允许打断
- `duplex.end_session_words`：结束会话口令
- `services.*_dir`：三个底层项目的路径

## 使用

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

进入语音对话：

```powershell
uv run chatcaht chat
```

无声模式：

```powershell
uv run chatcaht chat --no-audio
```

## 说明

ChatCaht 本身不重复实现唤醒词、ASR 或 TTS 的底层模型，而是通过它们暴露的协议做编排。这样每个子项目都可以独立训练、升级和调试，而 ChatCaht 负责整体会话状态、服务管理、LM Studio 对接和全链路 CLI。

## 许可证

MIT
