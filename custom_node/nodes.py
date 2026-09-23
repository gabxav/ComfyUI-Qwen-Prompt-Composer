"""Automatic thinking prompt composition on vLLM with a confirmed VRAM release barrier.

Flow: ComfyUI unloads its models -> lease on the vLLM pod's gpu_control wakes the model ->
streamed generation with thinking -> release with sleep=true suspends vLLM (sleep level 1 +
cuda-checkpoint) -> only then is the prompt returned, so the next nodes get the GPU.
SGLangPromptComposer stays registered as a hidden alias so older workflows still load.
"""
import json
import logging
import os
import urllib.error
import urllib.request

import comfy.model_management as mm
from comfy.utils import ProgressBar
from .core import build_messages, selected_slot

LOG = logging.getLogger(__name__)
DEFAULT_URL = os.getenv('COMFYUI_LLM_CONTROL_URL', 'http://llm-qwen38-w4a16-control.ialab.svc.cluster.local:8001')
API_URL = os.getenv('COMFYUI_LLM_URL', 'http://llm-qwen38-w4a16.ialab.svc.cluster.local').rstrip('/')
MODEL = os.getenv('COMFYUI_LLM_MODEL', 'llm-qwen38-w4a16')
# vLLM --limit-mm-per-prompt (MM_IMAGES in llm-qwen38-w4a16/kubernetes/statefulset.yml).
MAX_IMAGES = int(os.getenv('COMFYUI_LLM_MAX_IMAGES', '40'))
LEGACY_URLS = ('http://sglang-qwen38.ialab.svc.cluster.local', 'http://10.150.14.59')


def control_url(value):
    value = (value or '').strip().rstrip('/')
    if not value or value in LEGACY_URLS:
        # Saved workflows still carry the retired SGLang controller.
        return DEFAULT_URL.rstrip('/')
    return value


def request(base, path, body=None, timeout=15):
    raw = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=raw, method='GET' if body is None else 'POST',
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors='replace')[:600]
        try:
            detail = json.loads(detail).get('error') or detail
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f'vLLM gpu_control HTTP {exc.code}: {detail}') from exc


def stream_completion(payload):
    """Streams so a ComfyUI interrupt closes the connection, which aborts the request in vLLM."""
    req = urllib.request.Request(API_URL + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                 method='POST', headers={'Content-Type': 'application/json'})
    try:
        response = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'vLLM HTTP {exc.code}: {exc.read().decode(errors="replace")[:600]}') from exc
    text, finish, usage = [], None, None
    with response:
        for line in response:
            mm.throw_exception_if_processing_interrupted()
            line = line.strip()
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            chunk = json.loads(data)
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                if delta.get('content'):
                    text.append(delta['content'])
                finish = choice.get('finish_reason') or finish
    answer = ''.join(text).strip()
    if not answer:
        raise RuntimeError('The model returned no final answer'
                           + (' (thinking exhausted max_tokens).' if finish == 'length' else '.'))
    if finish == 'length':
        LOG.warning('Composer answer was cut by max_tokens')
    return answer, usage


class QwenPromptComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'prompt': ('STRING', {'multiline': True, 'default': ''}),
            'active_prompt': ('STRING', {'default': 'system_prompt_1'}),
            'controller_url': ('STRING', {'default': DEFAULT_URL}),
            'max_tokens': ('INT', {'default': 24576, 'min': 1024, 'max': 32768}),
            'thinking_budget': ('INT', {'default': 16384, 'min': 256, 'max': 16384}),
            'temperature': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 2.0, 'step': 0.01}),
            'top_p': ('FLOAT', {'default': 0.95, 'min': 0.01, 'max': 1.0, 'step': 0.01}),
            'top_k': ('INT', {'default': 20, 'min': 0, 'max': 1000}),
            'seed': ('INT', {'default': 1, 'min': 0, 'max': 0x7fffffff}),
            'video_fps': ('FLOAT', {'default': 24.0, 'min': 0.1, 'max': 240.0}),
            'video_sample_fps': ('FLOAT', {'default': 1.0, 'min': 0.1, 'max': 24.0}),
            'max_video_frames': ('INT', {'default': 32, 'min': 2, 'max': 64}),
            'max_image_side': ('INT', {'default': 1536, 'min': 256, 'max': 4096, 'step': 64}),
        }, 'optional': {
            **{f'system_prompt_{i}': ('STRING', {'forceInput': True}) for i in range(1, 17)},
            **{f'image_{i}': ('IMAGE',) for i in range(1, 9)},
            'video': ('IMAGE',),
        }}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('generated_text',)
    FUNCTION = 'compose'
    CATEGORY = 'text/LLM'
    DESCRIPTION = ('Always thinking. Wakes the vLLM Qwen3.8, uses the selected external system prompt '
                   'and puts the model back to sleep (no VRAM) before returning.')

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A cached node must never bypass the memory handoff side effect.
        return float('nan')

    def compose(self, prompt, active_prompt, controller_url, max_tokens, thinking_budget,
                temperature, top_p, top_k, seed, video_fps, video_sample_fps,
                max_video_frames, max_image_side, **inputs):
        selected_slot(active_prompt)
        if max_tokens < thinking_budget + 512:
            raise ValueError('max_tokens must reserve at least 512 tokens beyond thinking_budget.')
        messages, reference_info = build_messages(prompt, active_prompt, inputs, video_fps,
                                                  video_sample_fps, max_video_frames, max_image_side)
        if reference_info['images'] + reference_info['video_frames'] > MAX_IMAGES:
            raise ValueError(f'vLLM accepts at most {MAX_IMAGES} images per request (reference images plus '
                             'sampled video frames). Reduce references, video_sample_fps or max_video_frames.')
        payload = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens, 'stream': True,
                   'stream_options': {'include_usage': True}, 'thinking_token_budget': thinking_budget,
                   'chat_template_kwargs': {'enable_thinking': True},
                   'temperature': temperature, 'top_p': top_p, 'top_k': top_k, 'seed': seed}
        if len(json.dumps(payload).encode()) > 48 * 1024 * 1024:
            raise ValueError('References exceed 48 MiB. Reduce their resolution or frame count.')
        control = control_url(controller_url)
        mm.throw_exception_if_processing_interrupted()
        progress = ProgressBar(4)
        mm.unload_all_models()
        mm.soft_empty_cache()
        progress.update_absolute(1)
        mm.throw_exception_if_processing_interrupted()
        # ComfyUI already freed its own models above; gpu_control only waits for the VRAM.
        lease = request(control, '/acquire', {'owner': 'comfyui-composer', 'ttl': 2400, 'free_comfyui': False},
                        timeout=960)['lease']
        answer, released = None, False
        try:
            progress.update_absolute(2)
            answer, usage = stream_completion(payload)
            progress.update_absolute(3)
        finally:
            # Always put the model back to sleep, even on failure or cancel; waits for other leases.
            try:
                state = request(control, '/release', {'lease': lease, 'sleep': True}, timeout=1200)
                released = state.get('state') == 'suspended'
                if not released:
                    LOG.error('vLLM did not confirm suspension: %s', state)
            except Exception:
                LOG.exception('Could not put vLLM back to sleep; do not run GPU workflows until it is suspended.')
        if not released:
            raise RuntimeError('vLLM did not confirm it released the GPU. The prompt was withheld; '
                               'check gpu_control before running GPU workflows.')
        progress.update_absolute(4)
        LOG.info('vLLM prompt ready after VRAM release: %s; %s', reference_info, usage)
        return {'ui': {'text': [answer]}, 'result': (answer,)}


class SGLangPromptComposer(QwenPromptComposer):
    """Name used before 22/09/2026 (SGLang backend); hidden from the node menu."""
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {'QwenPromptComposer': QwenPromptComposer, 'SGLangPromptComposer': SGLangPromptComposer}
NODE_DISPLAY_NAME_MAPPINGS = {'QwenPromptComposer': 'Qwen Prompt Composer',
                              'SGLangPromptComposer': 'Qwen Prompt Composer (nome antigo)'}
