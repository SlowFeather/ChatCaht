# ChatCaht stack

`ChatCaht` owns stack orchestration while WakeUp, SpText, GVoice, and LoLLama remain independent repositories and environments.

```powershell
uv run chatcaht assets verify
uv run chatcaht assets download
uv run chatcaht services start
uv run chatcaht doctor
```

`assets verify` checks the deployed files against `models.manifest.yaml`. `assets download` only runs a component's provision command when its files are missing or invalid. The custom WakeUp model has no public download command and must be supplied from the project's release or training output.

Track `configs/config.example.yaml`, keep `configs/config.yaml` local, and pass secrets through `CHATCAHT_OPENAI_API_KEY`, `LOLLAMA_UPSTREAM_API_KEY`, `LOLLAMA_EMBEDDING_API_KEY`, or standard provider variables such as `HF_TOKEN`.
