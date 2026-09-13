import asyncio
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('controller', Path(__file__).parents[1] / 'controller/controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def result(text='Final prompt', finish='stop', reasoning='thinking'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': text, 'reasoning_content': reasoning}}]}


def test_no_reasoning_or_truncated_output_escapes():
    for data in [result(finish='length'), result(reasoning=''), result(text=''), result(text='<think>hidden')]:
        with pytest.raises(ValueError):
            c.final_answer(data)
    assert c.final_answer(result()) == 'Final prompt'


def test_thinking_is_forced_and_final_space_reserved():
    data = {'messages': [{'role': 'system', 'content': 'A'}, {'role': 'user', 'content': 'B'}],
            'model': 'other', 'stream': True, 'chat_template_kwargs': {'enable_thinking': False}}
    r = c.validate_payload(data)
    assert r['chat_template_kwargs']['enable_thinking'] is True
    assert r['model'] == c.MODEL and r['stream'] is False
    with pytest.raises(ValueError):
        c.validate_payload({**data, 'thinking_budget': 16384, 'max_tokens': 16384})


def test_output_waits_for_memory_release():
    async def scenario():
        r = c.Runtime(); r.phase = 'asleep'
        await r.lock.acquire()
        released = asyncio.Event(); suspended = asyncio.Event()
        async def wake(): r.phase = 'awake'
        async def post(*args, **kwargs): return result()
        async def suspend():
            suspended.set(); await released.wait(); r.phase = 'asleep'; r.last_memory = {'free_mib': 31495}
        r.wake = wake; r.post = post; r.suspend = suspend
        job = {'status': 'running'}
        task = asyncio.create_task(r.run_job(job, {}))
        await suspended.wait()
        assert job['status'] == 'running' and 'text' not in job and r.lock.locked()
        released.set(); await task
        assert job['text'] == 'Final prompt' and not r.lock.locked()
    asyncio.run(scenario())


def test_failed_release_does_not_publish_text():
    async def scenario():
        r = c.Runtime(); r.phase = 'awake'; await r.lock.acquire()
        async def wake(): pass
        async def post(*args, **kwargs): return result()
        async def suspend(): raise RuntimeError('CUDA failed')
        r.wake = wake; r.post = post; r.suspend = suspend
        job = {'status': 'running'}; await r.run_job(job, {})
        assert job['status'] == 'error' and 'text' not in job and r.phase == 'failed'
    asyncio.run(scenario())


def test_cancel_aborts_and_releases_memory():
    async def scenario():
        r = c.Runtime(); r.phase = 'awake'; await r.lock.acquire(); generating = asyncio.Event(); calls = []
        async def wake(): pass
        async def post(path, *args, **kwargs):
            calls.append(path)
            if path == '/v1/chat/completions':
                generating.set(); await asyncio.Event().wait()
        async def suspend(): calls.append('suspend'); r.phase = 'asleep'
        r.wake = wake; r.post = post; r.suspend = suspend
        job = {'status': 'running'}; task = asyncio.create_task(r.run_job(job, {}))
        await generating.wait(); task.cancel(); await task
        assert calls[-2:] == ['/abort_request', 'suspend'] and job['status'] == 'error'
        assert r.phase == 'asleep' and not r.lock.locked()
    asyncio.run(scenario())


def test_cancel_during_wake_never_starts_inference():
    async def scenario():
        r = c.Runtime(); r.phase = 'asleep'; await r.lock.acquire(); calls = []
        async def wake(): r.phase = 'awake'
        async def post(path, *args, **kwargs): calls.append(path)
        async def suspend(): r.phase = 'asleep'
        r.wake = wake; r.post = post; r.suspend = suspend
        job = {'status': 'running', 'cancel_requested': True}; await r.run_job(job, {})
        assert '/v1/chat/completions' not in calls and job['status'] == 'error' and r.phase == 'asleep'
    asyncio.run(scenario())
