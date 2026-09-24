# Local Gemma 8731 — approved canonical-template rollout, 2026-09-13

The user explicitly approved this restart ("해도된다."). Only the existing port-8731
Gemma process was restarted. GPU0 RTX4090, NF4 4-bit, single-device and checkpoint
`5926caa4ec0cac5cbfadaf4077420520de1d5205` were retained. No A100, new checkpoint,
system-driver replacement, GPU1 job or survey-service change was made.

Current module: `webchat_app_template_20260913:app`. Original `webchat_app.py` is
unchanged and remains the rollback module. PID at rollout was 1832383, replacing
1741207; the two local PID files were updated, but always revalidate `/health`,
the actual process command and port before using a PID.

Do not use `start_webchat.sh` or `start_dual_int8.sh` as the canonical restart command:
those historical scripts select the original module and different default GPU/precision.
They are preserved, not silently repurposed. No systemd unit/autostart was installed.

Canonical launch command, **only when port 8731 is free and a restart is authorized**:

```bash
env CUDA_VISIBLE_DEVICES=0 GEMMA4_12B_GPU=0 GEMMA4_12B_DEVICE_MAP=single GEMMA4_12B_LOAD_MODE=4bit \
  LD_LIBRARY_PATH=/home/iris/workspace/TREET/runtime/assets/nvidia-535.309/root/usr/lib/x86_64-linux-gnu \
  /home/iris/.local/share/mamba/envs/localllm-chat/bin/uvicorn \
  --app-dir /home/iris/workspace/localLLM/gemma4/12B webchat_app_template_20260913:app \
  --host 127.0.0.1 --port 8731
```

For rollback after validating and gracefully stopping the exact new process, use the
same command with `webchat_app:app`. This restores the previous application, **not**
a correct-template model-quality baseline. No rollback was needed during this rollout.
The driver path above points to the very same library/inode used by the previous
process; its old environment variable still named the removed MedicalAOS directory.

Active template dependency (do not remove during artifact cleanup):
`/home/iris/workspace/TREET/artifacts/experiments/treet_gemma4_template_20260913_v1/official_chat_template.jinja`.
It is hash-checked on app import and assigned to the local processor, not written to
the Hugging Face model cache. New responses add `raw_reply`, template/source hashes
and the actual serialized prompt hash; existing reply parsing/generation settings remain.

Rollout records/log:
`/home/iris/workspace/TREET/artifacts/experiments/treet_gemma4_rollout_20260913_v1/`.
`preflight.json` covers 640 historical prompts and 8 edge cases without model weights.
`application.json` records the startup handoff; its "awaiting format check" state is a
frozen startup snapshot, not the latest experiment status.

Completed 12-request F2 check:
`/home/iris/workspace/TREET/artifacts/experiments/treet_template_recheck_20260913_v1/summary.json`.
All requests had canonical prompt/source receipts, and three health observations
span 60.002 seconds. Format suitability remained 2/6 in each condition (same as F1).
This is transport/serialization verification, not clinical recovery or SCC efficacy.
