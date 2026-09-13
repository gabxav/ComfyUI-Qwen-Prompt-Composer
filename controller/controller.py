"""SGLang gateway and exclusive, fail-closed GPU handoff for ComfyUI."""
import asyncio
import ctypes
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

LOG = logging.getLogger('composer')
logging.basicConfig(level=logging.INFO)
BACKEND = 'http://127.0.0.1:8001'
MODEL = os.getenv('COMPOSER_MODEL', 'qwen3.8-27b-uncensored-dspark')
MAX_BODY = 48 * 1024 * 1024


class CUDA:
    class LockArgs(ctypes.Structure):
        _fields_ = [('timeoutMs', ctypes.c_uint), ('reserved0', ctypes.c_uint),
                    ('reserved1', ctypes.c_uint64 * 7)]

    def __init__(self):
        self.lib = ctypes.CDLL('libcuda.so.1')
        self.call('cuInit', 0)

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            error = ctypes.c_char_p()
            self.lib.cuGetErrorString(result, ctypes.byref(error))
            raise RuntimeError(f'{name}: CUDA {result}: {error.value.decode()}')

    def state(self, pid):
        value = ctypes.c_int()
        self.call('cuCheckpointProcessGetState', pid, ctypes.byref(value))
        return value.value

    def checkpoint(self, pid):
        args = self.LockArgs()
        args.timeoutMs = 5000
        self.call('cuCheckpointProcessLock', pid, ctypes.byref(args))
        try:
            self.call('cuCheckpointProcessCheckpoint', pid, None)
            if self.state(pid) != 2:
                raise RuntimeError('CUDA checkpoint state was not confirmed')
        except Exception:
            if self.state(pid) == 1:
                self.call('cuCheckpointProcessUnlock', pid, None)
            raise

    def restore(self, pid):
        if self.state(pid) != 2:
            raise RuntimeError('Worker is not checkpointed; refusing an unsafe restore')
        self.call('cuCheckpointProcessRestore', pid, None)
        self.call('cuCheckpointProcessUnlock', pid, None)
        if self.state(pid) != 0:
            raise RuntimeError('CUDA running state was not confirmed')


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    used, free = [int(x.strip()) for x in raw.splitlines()[0].split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                         '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {'used_mib': used, 'free_mib': free, 'processes': processes.strip()}


def scheduler_pid():
    # Only processes in this container's PID namespace are considered.
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / 'comm').read_text().strip().startswith('sglang::schedul'):
                matches.append(int(entry.name))
        except (OSError, ProcessLookupError):
            pass
    if len(matches) != 1:
        raise RuntimeError(f'Expected one SGLang scheduler, found {len(matches)}')
    return matches[0]


def validate_payload(data):
    if not isinstance(data, dict) or not isinstance(data.get('messages'), list):
        raise ValueError('messages must be a list')
    if len(data['messages']) != 2 or [m.get('role') for m in data['messages']] != ['system', 'user']:
        raise ValueError('Expected one system message and one user message')
    total = data.get('max_tokens', 24576)
    budget = data.pop('thinking_budget', 16384)
    if not isinstance(total, int) or not isinstance(budget, int) or not 256 <= budget <= 16384 or not budget + 512 <= total <= 32768:
        raise ValueError('Reserve at least 512 final tokens beyond the thinking budget (max total 32768)')
    allowed = {'messages', 'max_tokens', 'temperature', 'top_p', 'top_k', 'min_p',
               'repetition_penalty', 'presence_penalty', 'seed'}
    result = {k: v for k, v in data.items() if k in allowed}
    result.update(model=MODEL, max_tokens=total, stream=False,
                  chat_template_kwargs={'enable_thinking': True},
                  custom_params={'thinking_budget': budget})
    return result


def final_answer(data):
    choice = data['choices'][0]
    message = choice['message']
    text = message.get('content') or ''
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Generation did not finish. Increase the output budget or shorten the input.')
    if not message.get('reasoning_content'):
        raise ValueError('The backend did not return a separate thinking channel.')
    if not text.strip() or '<think>' in text or '</think>' in text:
        raise ValueError('No clean final answer was returned; reasoning is never used as a fallback.')
    return text.strip()


class Runtime:
    def __init__(self):
        self.phase = 'starting'
        self.lock = asyncio.Lock()
        self.jobs = {}
        self.pid = None
        self.models = None
        self.child = None
        self.cuda = None
        self.client = None
        self.failure = None
        self.last_memory = None

    async def post(self, path, body, timeout=120):
        response = await self.client.post(BACKEND + path, json=body, timeout=timeout)
        response.raise_for_status()
        return response.json() if response.content else None

    async def suspend(self):
        self.phase = 'suspending'
        await self.post('/pause_generation', {'mode': 'retract'})
        await self.post('/release_memory_occupation', {})
        self.pid = await asyncio.to_thread(scheduler_pid)
        await asyncio.to_thread(self.cuda.checkpoint, self.pid)
        self.last_memory = await asyncio.to_thread(gpu_snapshot)
        if any(line.split(',')[0].strip() == str(self.pid) for line in self.last_memory['processes'].splitlines()):
            raise RuntimeError('Worker still occupies GPU memory after checkpoint')
        self.phase = 'asleep'

    async def wake(self):
        if self.phase == 'awake':
            return
        if self.phase != 'asleep':
            raise RuntimeError(f'Cannot wake while state is {self.phase}')
        snapshot = await asyncio.to_thread(gpu_snapshot)
        # The tested 110K NVFP4 configuration requires about 30 GiB on this GPU.
        required = int(os.getenv('COMPOSER_MIN_FREE_MIB', '30500'))
        if snapshot['free_mib'] < required:
            raise ValueError(f"Only {snapshot['free_mib']} MiB free; need {required}. Unload ComfyUI models first.")
        if self.pid != await asyncio.to_thread(scheduler_pid):
            raise RuntimeError('Worker PID changed while sleeping')
        self.phase = 'waking'
        await asyncio.to_thread(self.cuda.restore, self.pid)
        await self.post('/resume_memory_occupation', {})
        await self.post('/continue_generation', {'torch_empty_cache': False})
        self.phase = 'awake'

    async def run_job(self, job, payload):
        started = time.monotonic()
        text = None
        error = None
        try:
            job['stage'] = 'waking'
            await self.wake()
            if job.get('cancel_requested'):
                raise asyncio.CancelledError()
            job['stage'] = 'thinking'
            data = await self.post('/v1/chat/completions', payload, timeout=1800)
            text = final_answer(data)
            job['usage'] = data.get('usage')
        except asyncio.CancelledError:
            error = 'Cancelled by ComfyUI'
            if self.phase == 'awake':
                await self.post('/abort_request', {'abort_all': True})
        except Exception as exc:
            error = str(exc)
            if self.phase == 'awake':
                try:
                    await self.post('/abort_request', {'abort_all': True})
                except Exception:
                    LOG.exception('Abort failed')
        finally:
            job['stage'] = 'suspending'
            try:
                if self.phase == 'awake':
                    await self.suspend()
                elif self.phase != 'asleep':
                    raise RuntimeError(f'Incomplete memory transition: {self.phase}')
                job['memory'] = self.last_memory
            except Exception as exc:
                self.phase = 'failed'
                self.failure = f'Memory handoff failed: {exc}'
                error = self.failure
                LOG.exception('Memory handoff failed; GPU output barrier remains closed')
            job['seconds'] = round(time.monotonic() - started, 3)
            # Only publish a final answer after physical GPU release is confirmed.
            if job.get('cancel_requested'):
                error = error or 'Cancelled by ComfyUI'
            if error:
                job.update(status='error', error=error, stage='error')
            else:
                job.update(status='complete', text=text, stage='complete')
            job['finished'] = time.time()
            self.lock.release()

    async def initialize(self):
        self.cuda = await asyncio.to_thread(CUDA)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if self.child.poll() is not None:
                raise RuntimeError('SGLang exited during startup')
            try:
                r = await self.client.get(BACKEND + '/health_generate', timeout=5)
                r.raise_for_status()
                self.models = (await self.client.get(BACKEND + '/v1/models')).json()
                self.phase = 'awake'
                # Start with VRAM available to ComfyUI, including after a restart.
                async with self.lock:
                    await self.suspend()
                return
            except (httpx.HTTPError, OSError):
                await asyncio.sleep(2)
        raise RuntimeError('SGLang startup timed out')

    async def monitor(self):
        try:
            await self.initialize()
            while True:
                await asyncio.sleep(5)
                if self.child.poll() is not None:
                    raise RuntimeError('SGLang subprocess exited')
                now = time.time()
                for key, job in list(self.jobs.items()):
                    if job.get('finished', now) < now - 600:
                        self.jobs.pop(key, None)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.phase = 'failed'
            self.failure = str(exc)
            LOG.exception('Controller health failed')


runtime = Runtime()


@asynccontextmanager
async def lifespan(app):
    args = sys.argv[1:]
    for flag, value in [('--port', '8001'), ('--host', '127.0.0.1')]:
        if flag in args:
            args[args.index(flag) + 1] = value
        else:
            args.extend([flag, value])
    runtime.child = subprocess.Popen(['sglang', 'serve', *args], start_new_session=True)
    runtime.client = httpx.AsyncClient(timeout=30)
    monitor = asyncio.create_task(runtime.monitor())
    yield
    monitor.cancel()
    for job in runtime.jobs.values():
        task = job.get('task')
        if task and not task.done():
            task.cancel()
    # Container termination intentionally discards the RAM checkpoint.
    if runtime.child.poll() is None:
        os.killpg(runtime.child.pid, signal.SIGTERM)
    await runtime.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get('/health')
@app.get('/health_generate')
async def health():
    if runtime.phase in ('starting', 'failed') or runtime.child is None or runtime.child.poll() is not None:
        return JSONResponse({'state': runtime.phase, 'error': runtime.failure}, status_code=503)
    if runtime.phase == 'asleep':
        try:
            if await asyncio.to_thread(runtime.cuda.state, runtime.pid) != 2:
                raise RuntimeError('Unexpected checkpoint state')
        except Exception:
            return JSONResponse({'state': 'failed'}, status_code=503)
    return {'state': runtime.phase}


@app.get('/composer/status')
async def status():
    return {'state': runtime.phase, 'busy': runtime.lock.locked(), 'memory': runtime.last_memory,
            'error': runtime.failure, 'model': MODEL, 'context_length': 110000,
            'thinking_budget_max': 16384}


@app.post('/composer/jobs')
async def submit(request: Request):
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise HTTPException(413, 'Reference payload exceeds 48 MiB')
    try:
        payload = validate_payload(json.loads(raw))
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if runtime.phase not in ('awake', 'asleep') or runtime.lock.locked():
        raise HTTPException(409, 'SGLang is busy or unavailable; retry when idle')
    await runtime.lock.acquire()
    key = uuid.uuid4().hex
    job = {'id': key, 'status': 'running', 'stage': 'queued', 'created': time.time()}
    runtime.jobs[key] = job
    job['task'] = asyncio.create_task(runtime.run_job(job, payload))
    return {'id': key, 'status': 'running'}


@app.get('/composer/jobs/{key}')
async def get_job(key: str):
    job = runtime.jobs.get(key)
    if not job:
        raise HTTPException(404, 'Job not found or expired')
    return {k: v for k, v in job.items() if k != 'task'}


@app.delete('/composer/jobs/{key}')
async def cancel(key: str):
    job = runtime.jobs.get(key)
    if not job:
        raise HTTPException(404, 'Job not found')
    # Do not interrupt wake/suspend, where cancellation could strand the driver.
    if job['status'] == 'running' and not job.get('cancel_requested'):
        job['cancel_requested'] = True
        if job['stage'] == 'thinking':
            job['task'].cancel()
    return {'status': 'cancellation_requested'}


@app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def proxy(path: str, request: Request):
    if path == 'v1/models' and request.method == 'GET' and runtime.models:
        return runtime.models
    readonly = path in ('metrics', 'model_info', 'get_model_info') and request.method == 'GET'
    # Administrative memory endpoints must not bypass the controller's state machine.
    allowed = path in ('v1/chat/completions', 'v1/completions', 'v1/tokenize', 'tokenize',
                       'v1/detokenize', 'detokenize', 'v1/embeddings', 'get_server_info', 'server_info')
    if not readonly and not allowed:
        raise HTTPException(404, 'Endpoint is not exposed by the memory controller')
    if not readonly and (runtime.phase not in ('awake', 'asleep') or runtime.lock.locked()):
        raise HTTPException(423, 'GPU is reserved or SGLang is unavailable')
    acquired = False
    try:
        if not readonly:
            await runtime.lock.acquire()
            acquired = True
            await runtime.wake()
        body = await request.body()
        url = BACKEND + '/' + path
        if request.url.query:
            url += '?' + request.url.query
        headers = {k: v for k, v in request.headers.items() if k.lower() in ('content-type', 'accept', 'authorization')}
        req = runtime.client.build_request(request.method, url, content=body, headers=headers, timeout=1800)
        response = await runtime.client.send(req, stream=True)
        async def stream():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                if acquired:
                    async def cleanup():
                        try:
                            await runtime.post('/abort_request', {'abort_all': True})
                            await runtime.suspend()
                        except Exception as exc:
                            runtime.phase = 'failed'
                            runtime.failure = str(exc)
                            LOG.exception('Proxy cleanup failed')
                        finally:
                            runtime.lock.release()
                    await asyncio.shield(asyncio.create_task(cleanup()))
        return StreamingResponse(stream(), status_code=response.status_code,
                                 media_type=response.headers.get('content-type'))
    except Exception:
        if acquired:
            try:
                if runtime.phase == 'awake':
                    await runtime.suspend()
            except Exception as exc:
                runtime.phase = 'failed'
                runtime.failure = str(exc)
            finally:
                runtime.lock.release()
        raise


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
