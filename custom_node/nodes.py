"""Automatic thinking prompt composition on the vLLM Qwen3.8 resident on its dedicated GPU.

Flow: streamed generation with thinking straight from /v1/chat/completions; ComfyUI keeps its models loaded.
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
API_URL = os.getenv('COMFYUI_LLM_URL', 'http://llm-qwen38-27b.ialab.svc.cluster.local').rstrip('/')
MODEL = os.getenv('COMFYUI_LLM_MODEL', 'llm-qwen38-27b')
# vLLM --limit-mm-per-prompt (MM_IMAGES in llm-qwen38-27b/kubernetes/statefulset.yml).
MAX_IMAGES = int(os.getenv('COMFYUI_LLM_MAX_IMAGES', '40'))


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
            # Unused; kept in place so saved workflows keep their widget order.
            'controller_url': ('STRING', {'default': '', 'tooltip': 'Obsolete, ignored.'}),
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
    DESCRIPTION = ('Always thinking. Uses the vLLM Qwen3.8 with the selected external system prompt '
                   'on a dedicated GPU without unloading ComfyUI models.')

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute when explicitly queued, preserving the node's existing cache policy.
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
        # The LLM runs on a dedicated GPU, so neither ComfyUI nor vLLM needs unloading.
        mm.throw_exception_if_processing_interrupted()
        progress = ProgressBar(1)
        answer, usage = stream_completion(payload)
        progress.update_absolute(1)
        LOG.info('vLLM prompt ready (dedicated GPU): %s; %s', reference_info, usage)
        return {'ui': {'text': [answer]}, 'result': (answer,)}


class SGLangPromptComposer(QwenPromptComposer):
    """Name used before 22/09/2026 (SGLang backend); hidden from the node menu."""
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {'QwenPromptComposer': QwenPromptComposer, 'SGLangPromptComposer': SGLangPromptComposer}
NODE_DISPLAY_NAME_MAPPINGS = {'QwenPromptComposer': 'Qwen Prompt Composer',
                              'SGLangPromptComposer': 'Qwen Prompt Composer (nome antigo)'}
