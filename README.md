# ComfyUI Qwen Prompt Composer

A multimodal prompt-writing node for ComfyUI with editable external system prompts and always-on thinking. It streams directly from a resident vLLM server on a dedicated GPU and keeps ComfyUI models loaded.

Formerly **ComfyUI SGLang Prompt Composer**. Since 2026-09-22 it runs on vLLM; the old class name `SGLangPromptComposer` stays registered (hidden from the menu) so saved workflows still load, and the obsolete `controller_url` input remains in its original position but is ignored.

## Features

- Select one external system prompt by the title of its connected text node.
- Dynamic sockets: connecting the last system prompt or image reveals the next input. Disconnecting trims unused trailing inputs without renaming existing connections.
- Up to 16 system prompt inputs, 8 image inputs, and one video frame batch.
- Ordered image references and sampled video frames with timestamps.
- Streams the answer, so a ComfyUI cancel aborts the request on the server.
- Returns the final answer without reasoning text or changing model residency.

## Requirements

- A resident vLLM server exposing OpenAI-compatible streaming `/v1/chat/completions` and supporting the configured reasoning model and multimodal inputs. The XavierLAB installation uses `llm-qwen38-27b` on a dedicated GPU.
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
| `COMFYUI_LLM_URL` | `http://llm-qwen38-27b.ialab.svc.cluster.local` | vLLM OpenAI-compatible base URL, without `/v1` |
| `COMFYUI_LLM_MODEL` | `llm-qwen38-27b` | served model name |
| `COMFYUI_LLM_MAX_IMAGES` | `40` | must match the server's image limit |

## Resident-model execution

The node sends requests directly to vLLM and returns the streamed answer. It does not unload ComfyUI models, acquire GPU leases, or call sleep/wake endpoints. Cancellation closes the stream without changing model residency. The `controller_url` widget is retained only for compatibility with saved workflows; its value is ignored.

The earlier GPU handoff controller in `server/gpu_control.py` and its tests are retained as legacy utilities. They are not used or required by the current node. Deployments that depend on automatic GPU handoff should retain the earlier implementation rather than expect that behavior from this version.

## Related nodes and source

This repository is the canonical source. The duplicate `ialab/comfyui-qwen-composer` folder in the XavierLAB GitOps repository was retired after its resident-model implementation was synchronized here.

The separate `ComfyUI-Qwen-Prompt-Composer-H3` and `ComfyUI-Qwen-Prompt-Composer-Generic` packages call the Prompt Enhancer API and maintain their own repositories. The laboratory's former `prompt_creator_qwen` workflow was archived on 2026-09-22 as `workflow-archive/prompt_creator_qwen-20260922.json`; the H3 workflow uses its dedicated node, and other prompt apps use Generic.

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

Historical validation of the former GPU handoff implementation on 2026-09-22: a text-only workflow completed in 20 s and a three-image workflow in 40 s, both starting from a suspended model, and GPU usage returned to the pre-run level (the vLLM process at 0). Decode speed after a suspend/restore cycle matched the speed before it. Heavy downstream image/video rendering and broad hardware/runtime compatibility remain to be evaluated.

Source consolidation on 2026-10-19: the runtime `custom_node/nodes.py` matches the installed resident-model implementation byte for byte. Tests cover direct streamed completion, ignored legacy controller URLs, cancellation and failures without unloading models, payload construction and dynamic input compatibility. No service restart is required to consolidate these source repositories.
