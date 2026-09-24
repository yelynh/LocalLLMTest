# Ollama Local Runtime

User-local Ollama install for the MedicalAOS third local LLM family.

## Paths

- Binary: `/home/iris/workspace/localLLM/ollama/root/bin/ollama`
- Models: `/home/iris/workspace/localLLM/ollama/models`
- Logs: `/home/iris/workspace/localLLM/ollama/logs/ollama.log`

## Start / Stop

```bash
/home/iris/workspace/localLLM/ollama/start_ollama.sh
/home/iris/workspace/localLLM/ollama/stop_ollama.sh
```

## Health

```bash
curl http://127.0.0.1:11434/api/version
OLLAMA_HOST=127.0.0.1:11434 ollama list
```

## Notes

- Installed from the official Linux amd64 tarball.
- This is user-local because system-wide install needs sudo password.
- Use this as the reproducible Llama/Ollama-family runtime control, separate from AUROC/AUPRC predictor baselines.
- Current canonical model: `llama3.1:8b-instruct-q8_0`.
- `llama3.2-vision:11b` was pulled but does not run on this Ollama runner because `mllama` is reported as an unknown architecture.
- GPU CUDA serving is disabled by default with `OLLAMA_LLM_LIBRARY=cpu` because the current NVIDIA 535 / CUDA 12.2 driver stack triggers `device kernel image is invalid` in Ollama's CUDA runner. Override `OLLAMA_LLM_LIBRARY` only after the driver/Ollama CUDA stack is fixed.
