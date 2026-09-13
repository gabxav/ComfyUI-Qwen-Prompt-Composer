"""Automatic thinking prompt composition with a confirmed VRAM release barrier."""
import json
import logging
import os
import time
import urllib.error
import urllib.request

import comfy.model_management as mm
from comfy.utils import ProgressBar
from .core import build_messages, selected_slot

LOG = logging.getLogger(__name__)
DEFAULT_URL = os.getenv('COMFYUI_LLM_URL', 'http://127.0.0.1:8000')


def request(base, path, body=None, method=None, timeout=15):
    raw = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip('/') + path, data=raw, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors='replace')[:600]
        raise RuntimeError(f'SGLang controller HTTP {exc.code}: {detail}') from exc


class SGLangPromptComposer:
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
    DESCRIPTION = 'Always thinking. Uses the selected external system prompt and releases SGLang VRAM before returning.'

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
        payload = {'messages': messages, 'max_tokens': max_tokens, 'thinking_budget': thinking_budget,
                   'temperature': temperature, 'top_p': top_p, 'top_k': top_k, 'seed': seed}
        if len(json.dumps(payload).encode()) > 48 * 1024 * 1024:
            raise ValueError('References exceed 48 MiB. Reduce their resolution or frame count.')
        mm.throw_exception_if_processing_interrupted()
        state = request(controller_url, '/composer/status')
        if state['busy'] or state['state'] not in ('awake', 'asleep'):
            raise RuntimeError(f"SGLang controller unavailable: {state['state']}")
        progress = ProgressBar(4)
        mm.unload_all_models()
        mm.soft_empty_cache()
        progress.update_absolute(1)
        mm.throw_exception_if_processing_interrupted()
        job = request(controller_url, '/composer/jobs', payload)
        key = job['id']
        deadline = time.monotonic() + 1900
        completed = False
        try:
            while time.monotonic() < deadline:
                mm.throw_exception_if_processing_interrupted()
                current = request(controller_url, '/composer/jobs/' + key)
                if current['status'] == 'error':
                    raise RuntimeError(current['error'])
                if current['status'] == 'complete':
                    completed = True
                    if current.get('memory', {}).get('processes'):
                        # The controller already verifies its own PID. Other consumers
                        # may exist; do not claim the whole GPU is empty.
                        LOG.info('Other GPU processes remain after SGLang suspension')
                    progress.update_absolute(4)
                    LOG.info('SGLang prompt ready after VRAM release: %s; %s', reference_info, current.get('usage'))
                    return {'ui': {'text': [current['text']]}, 'result': (current['text'],)}
                progress.update_absolute(3 if current['stage'] == 'suspending' else 2)
                time.sleep(0.5)
            raise TimeoutError('Prompt composition exceeded its deadline.')
        finally:
            if not completed:
                try:
                    request(controller_url, '/composer/jobs/' + key, method='DELETE')
                    # Keep the execution barrier until cleanup finishes, even on cancel.
                    cleanup_deadline = time.monotonic() + 120
                    while time.monotonic() < cleanup_deadline:
                        status = request(controller_url, '/composer/jobs/' + key)
                        if status['status'] != 'running':
                            break
                        time.sleep(0.5)
                    else:
                        LOG.error('Controller cleanup is still pending; do not run GPU workflows.')
                except Exception:
                    LOG.exception('Could not confirm controller cleanup')


NODE_CLASS_MAPPINGS = {'SGLangPromptComposer': SGLangPromptComposer}
NODE_DISPLAY_NAME_MAPPINGS = {'SGLangPromptComposer': 'SGLang Vision Prompt Composer'}
