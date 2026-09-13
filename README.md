# ComfyUI SGLang Prompt Composer

A multimodal prompt-writing node for ComfyUI with editable external system prompts, always-on thinking, and automatic GPU memory handoff through the included SGLang controller.

## Features

- Select one external system prompt by the title of its connected text node.
- Dynamic sockets: connecting the last system prompt or image reveals the next input. Disconnecting trims unused trailing inputs without renaming existing connections.
- Up to 16 system prompt inputs, 8 image inputs, and one video frame batch.
- Ordered image references and sampled video frames with timestamps.
- Returns only the final answer, after the controller confirms that SGLang has released GPU memory.
- Preserves model weights in system RAM between requests.
- Cancellation and errors trigger backend cleanup; truncated or missing final answers are rejected.

## Requirements

This is currently a specialized integration, not a client for an arbitrary SGLang endpoint. It requires the controller in `controller/`, a compatible SGLang build with memory saver and CPU weight backup, and NVIDIA CUDA process checkpoint support.

The implementation was exercised with a single RTX 5090, NVIDIA driver 615.71.09, a Qwen3.8 27B DSPARK NVFP4 SGLang build, a 110,000-token context and a 16,384-token thinking ceiling. Other GPUs, tensor-parallel deployments and runtime builds have not been validated. The controller expects exactly one scheduler process in its process namespace.

## Install the ComfyUI node

From your ComfyUI directory:

```bash
git clone https://github.com/gabxav/ComfyUI-SGLang-Prompt-Composer.git custom_nodes/ComfyUI-SGLang-Prompt-Composer
python -m pip install -r custom_nodes/ComfyUI-SGLang-Prompt-Composer/requirements.txt
```

Use the Python environment that runs ComfyUI. Restart ComfyUI while its queue is idle, then reopen the browser page. Find **SGLang Vision Prompt Composer** under `text/LLM`.

Import `workflows/prompt_composer.json` for a minimal example using built-in text nodes. Edit the system text and user request, then set `controller_url` to the controller's reachable base URL, without `/v1`. The default is `http://127.0.0.1:8000`; in containers, localhost refers to the ComfyUI container. `COMFYUI_LLM_URL` can override the node's default URL.

## Controller

Run the controller inside the environment containing your compatible SGLang installation and CUDA driver libraries:

```bash
python -m pip install -r controller/requirements.txt
export COMPOSER_MODEL='your-served-model-name'
python controller/controller.py \
  --model-path /models/your-model \
  --served-model-name "$COMPOSER_MODEL" \
  --context-length 110000 \
  --enable-memory-saver \
  --enable-weights-cpu-backup
```

This illustrates the wrapper, not a complete model-specific launch configuration. Retain your validated quantization, speculative decoding, draft-model, vision, thinking and allocator flags. A speculative draft may also need CPU weight backup. The exact SGLang build must support `pause_generation`, `release_memory_occupation`, `resume_memory_occupation`, `continue_generation`, `abort_request`, separated `reasoning_content`, and `custom_params.thinking_budget`.

The controller launches `sglang serve` on `127.0.0.1:8001` and exposes its gateway on port 8000. `COMPOSER_MODEL` must match the model's served name. `COMPOSER_MIN_FREE_MIB` defaults to 30,500 MiB, the threshold used for the tested 110K configuration; do not lower it without validating the actual memory requirement. Model/context defaults remain specific to that tested configuration.

The optional Dockerfile accepts your validated SGLang base image through the `SGLANG_BASE_IMAGE` build argument. It copies the controller to `/opt/composer/controller.py`; configure the container command accordingly. No runtime image is bundled or published by this repository.

The gateway has no built-in authentication. Keep it on a trusted network or behind your existing authenticated gateway. Administrative memory endpoints are not forwarded directly. The wrapper also handles selected `/v1` endpoints, serializes requests, and suspends after they finish; a complete regression of external clients has not been performed.

## GPU handoff

```text
ComfyUI unloads managed models and clears its allocator cache
→ controller checks free VRAM
→ restores CUDA checkpoint and SGLang memory
→ generates with thinking
→ pauses generation and releases SGLang memory
→ checkpoints the CUDA worker and verifies its GPU allocation is gone
→ returns the final prompt to downstream nodes
```

Weights stay in RAM while the process is suspended. Initial server startup or restart still requires a full model load. Other processes and live tensors can occupy VRAM; the node cannot release memory it does not own. Connect the output to downstream consumers to establish graph execution order. This is not a distributed GPU scheduler.

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
| `thinking_budget` | Default ceiling 16,384; thinking is always enabled. |
| `max_tokens` | Default 24,576, shared by thinking and final answer. |

The input payload is capped at 48 MiB. Explain reference roles such as identity, environment or first/last frame in your request. The transport preserves order but does not invent these roles. System prompt quality governs output style and structure; thinking alone does not guarantee quality.

## Tests and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest --confcutdir=tests tests
node --test tests/dynamic_inputs.test.mjs
```

The original deployment passed text generation, two-image plus sampled-frame inference, cancellation, and a subsequent successful request. Its copied workflow completed in 17.6 seconds including memory handoff. After suspension it reported about 30.75 GiB free; this is an observation for one configuration, not a benchmark guarantee. Dynamic connections, disconnection without renumbering, and graph save/reload were checked in the actual ComfyUI frontend.

Heavy downstream image/video rendering, long-term idle endurance and broad hardware/runtime compatibility remain to be evaluated.
