"""Exercise resident-model requests without requiring a ComfyUI installation."""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def resident(monkeypatch):
    package = types.ModuleType('resident_composer')
    package.__path__ = [str(Path(__file__).parents[1] / 'custom_node')]
    mm = types.ModuleType('comfy.model_management')
    mm.throw_exception_if_processing_interrupted = Mock()
    mm.unload_all_models = Mock(side_effect=AssertionError('Must keep ComfyUI loaded'))
    mm.soft_empty_cache = Mock(side_effect=AssertionError('Must preserve allocator cache'))
    comfy = types.ModuleType('comfy')
    comfy.model_management = mm
    utils = types.ModuleType('comfy.utils')
    utils.ProgressBar = Mock()
    for name, module in {'resident_composer': package, 'comfy': comfy,
                         'comfy.model_management': mm, 'comfy.utils': utils}.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ('core', 'nodes'):
        spec = importlib.util.spec_from_file_location('resident_composer.' + name,
                    Path(package.__path__[0]) / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module, mm


def compose(node):
    return node.QwenPromptComposer().compose(
        'Describe a landscape.', 'system_prompt_1', 'http://retired-controller.invalid',
        1024, 256, 0.6, 0.95, 20, 7, 24, 1, 32, 1536,
        system_prompt_1='Write one concise image prompt.')


def test_only_completion_endpoint_is_called_and_reasoning_is_not_returned(resident, monkeypatch):
    node, mm = resident
    chunks = [
        {'choices': [{'delta': {'reasoning_content': 'Internal analysis'}}]},
        {'choices': [{'delta': {'content': 'A green valley.'}, 'finish_reason': 'stop'}]},
    ]
    response = io.BytesIO(b''.join(b'data: ' + json.dumps(c).encode() + b'\n' for c in chunks)
                          + b'data: [DONE]\n')
    request = Mock(return_value=response)
    monkeypatch.setattr(node.urllib.request, 'urlopen', request)
    result = compose(node)
    assert result['result'] == ('A green valley.',)
    assert request.call_count == 1
    sent = request.call_args.args[0]
    assert sent.full_url == node.API_URL + '/v1/chat/completions'
    assert json.loads(sent.data)['chat_template_kwargs']['enable_thinking'] is True
    assert response.closed
    mm.unload_all_models.assert_not_called()
    mm.soft_empty_cache.assert_not_called()


@pytest.mark.parametrize('cancel', [False, True])
def test_errors_and_cancellation_propagate_without_gpu_handoff(resident, monkeypatch, cancel):
    node, mm = resident
    if cancel:
        mm.throw_exception_if_processing_interrupted.side_effect = RuntimeError('cancelled')
    stream = Mock(side_effect=RuntimeError('service unavailable'))
    monkeypatch.setattr(node, 'stream_completion', stream)
    with pytest.raises(RuntimeError, match='cancelled' if cancel else 'service unavailable'):
        compose(node)
    assert stream.call_count == (0 if cancel else 1)
    mm.unload_all_models.assert_not_called()
    mm.soft_empty_cache.assert_not_called()
