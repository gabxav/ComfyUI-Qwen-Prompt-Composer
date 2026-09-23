# ComfyUI Qwen Prompt Composer

A multimodal prompt-writing node for ComfyUI with editable external system prompts, always-on thinking, and automatic GPU memory handoff: it wakes a vLLM server, generates, and puts the server back to sleep with no VRAM before downstream nodes run.

Formerly **ComfyUI SGLang Prompt Composer**. Since 2026-09-22 it runs on vLLM; the old class name `SGLangPromptComposer` stays registered (hidden from the menu) so saved workflows still load, and an old SGLang `controller_url` saved in a workflow is redirected to the default below.

## Features

- Select one external system prompt by the title of its connected text node.
- Dynamic sockets: connecting the last system prompt or image reveals the next input. Disconnecting trims unused trailing inputs without renaming existing connections.
- Up to 16 system prompt inputs, 8 image inputs, and one video frame batch.
- Ordered image references and sampled video frames with timestamps.
- Streams the answer, so a ComfyUI cancel aborts the request on the server.
- Returns only the final answer, after `gpu_control` confirms vLLM has released the GPU. Weights stay in system RAM.

## Requirements

- A vLLM server started with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1` (for `/sleep`, `/wake_up`, `/is_sleeping`), serving a Qwen3-style reasoning model. Tested with vLLM 0.28.0, Qwen3.8 27B W4A16, an RTX 5090 and NVIDIA driver 615.71.09.
- `server/gpu_control.py` running in the same PID namespace as the vLLM engine (same container), with NVIDIA's [`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint) (driver 550+) and `nvidia-smi` available.
- The vLLM image limit (`--limit-mm-per-prompt`) must cover your references plus sampled video frames.

## Install the ComfyUI node

From your ComfyUI directory:

```bash
git clone https://github.com/gabxav/ComfyUI-Qwen-Prompt-Composer.git custom_nodes/ComfyUI-Qwen-Prompt-Composer
python -m pip install -r custom_nodes/ComfyUI-Qwen-Prompt-Composer/requirements.txt
```

Restart ComfyUI while its queue is idle, then reopen the browser page. Find **Qwen Prompt Composer** under `text/LLM`. Import `workflows/prompt_composer.json` for a minimal example.

| Variable | Default | Meaning |
| --- | --- | --- |
| `COMFYUI_LLM_CONTROL_URL` | `http://llm-qwen38-w4a16-control.ialab.svc.cluster.local:8001` | `gpu_control` base URL (also the node's `controller_url` default) |
| `COMFYUI_LLM_URL` | `http://llm-qwen38-w4a16.ialab.svc.cluster.local` | vLLM OpenAI-compatible base URL, without `/v1` |
| `COMFYUI_LLM_MODEL` | `llm-qwen38-w4a16` | served model name |
| `COMFYUI_LLM_MAX_IMAGES` | `40` | must match the server's image limit |

## GPU handoff

```text
ComfyUI unloads managed models and clears its allocator cache
→ POST /acquire on gpu_control: waits for free VRAM, wakes vLLM (restore + unlock + /wake_up, ~3 s)
→ streamed generation with thinking (thinking_token_budget)
→ POST /release with sleep=true: waits for other clients' leases, then /sleep?level=1 + cuda-checkpoint (~3 s)
→ returns the final prompt to downstream nodes
```

vLLM's sleep level 1 moves weights to RAM and drops the KV cache but leaves about 1.8 GiB (CUDA context, workspaces, CUDA graph pools); `cuda-checkpoint` then moves the engine's remaining CUDA state to host memory, so the vLLM process holds no VRAM. If the prompt cannot be confirmed released, the node withholds it and raises an error.

## gpu_control

`server/gpu_control.py` (Python standard library only) listens on port 8001 and hands out usage leases:

- `POST /acquire {owner, ttl?, free_comfyui?}` wakes the model if needed and returns a lease. Waking requires `WAKE_FREE_MIB` (24,500) free; when short and `free_comfyui` is true, an idle ComfyUI (`COMFYUI_URL`) is asked to `/free` first.
- `POST /release {lease, sleep?}` returns the lease; `sleep: true` waits up to 15 minutes for other leases, then suspends.
- `POST /suspend` (refused while leases are held), `POST /resume`, `GET /state`.

It has no authentication: restrict it to trusted clients (for example with a Kubernetes NetworkPolicy). Clients that call vLLM without a lease hang while it sleeps.

## Inputs

| Input | Behavior |
| --- | --- |
| `prompt` | User request; can be connected to an external text node. |
| `active_prompt` | Selects a connected system prompt by title. The socket identity remains stable when renamed. |
| `system_prompt_1…16` | Only the selected text is sent as the system message. |
| `image_1…8` | ComfyUI IMAGE tensors, including batches; at most 32 still images total. |
| `video` | IMAGE batch of frames; no native video file or audio stream. |
| `video_fps` | Effective frame rate of the incoming batch, default 24. Set it after any loader subsampling. |
| `video_sample_fps` | Requested sample rate, default 1 fps. |
| `max_video_frames` | Default 32, including the last frame; spreads samples across the video when capped. |
| `max_image_side` | Resizes references to at most 1536 pixels on the longest side by default. |
| `thinking_budget` | Default ceiling 16,384, sent as `thinking_token_budget`; thinking is always enabled. |
| `max_tokens` | Default 24,576, shared by thinking and final answer. |

The input payload is capped at 48 MiB. Explain reference roles such as identity, environment or first/last frame in your request. The transport preserves order but does not invent these roles. System prompt quality governs output style and structure; thinking alone does not guarantee quality.

## Tests and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest --confcutdir=tests tests
node --test tests/dynamic_inputs.test.mjs
```

Live validation on 2026-09-22: a text-only workflow completed in 20 s and a three-image workflow in 40 s, both starting from a suspended model, and GPU usage returned to the pre-run level (the vLLM process at 0). Decode speed after a suspend/restore cycle matched the speed before it. Heavy downstream image/video rendering and broad hardware/runtime compatibility remain to be evaluated.
