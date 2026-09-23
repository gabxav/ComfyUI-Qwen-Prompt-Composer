"""GPU suspend/resume with usage leases for the vLLM pod (port 8001, stdlib only).

vLLM's sleep level 1 frees weights and KV cache but leaves ~1.8 GiB (CUDA context,
workspaces, CUDA graph pools). cuda-checkpoint then moves the EngineCore's remaining
CUDA state to host RAM, so the process holds no VRAM at all.

Clients hold a lease while they use the model:
  POST /acquire {owner, ttl?, free_comfyui?}  wakes the model if needed -> {lease, woke}
  POST /release {lease, sleep?}               sleep=true (ComfyUI Composer) waits for the other
                                              leases, suspends and only then answers
Manual control (AI Chat Dormir/Acordar):
  POST /suspend  refused while leases are active     POST /resume
  GET  /state    awake | sleeping | suspended | suspending | resuming | locked | unavailable

Waking needs free VRAM. When it is short and free_comfyui is set, an idle ComfyUI is asked
to unload its models first. Requests to /v1 while suspended hang until a resume.
"""
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VLLM = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}"
CHECKPOINT = os.environ.get("CUDA_CHECKPOINT", "/usr/local/bin/cuda-checkpoint")
COMFYUI = os.environ.get("COMFYUI_URL", "http://comfyui-service.ialab.svc.cluster.local").rstrip("/")
# Measured 22/09/2026: the awake engine holds 23,961 MiB (24,669 total minus ComfyUI's 708).
WAKE_FREE_MIB = int(os.environ.get("WAKE_FREE_MIB", "24500"))
VRAM_WAIT = float(os.environ.get("VRAM_WAIT", "60"))
SLEEP_WAIT = float(os.environ.get("SLEEP_WAIT", "900"))
DEFAULT_TTL = 3600
MAX_TTL = 6 * 3600


class Busy(RuntimeError):
    """Retryable refusal (HTTP 409): ComfyUI busy, VRAM short, model in use."""


def engine_pids():
    # /proc/<pid>/comm is truncated to 15 characters.
    pids = []
    for comm in Path("/proc").glob("[0-9]*/comm"):
        try:
            if comm.read_text().strip().startswith("VLLM::EngineCor"):
                pids.append(int(comm.parent.name))
        except OSError:
            pass
    return sorted(pids)


def checkpoint(action, pid, *extra):
    args = [CHECKPOINT, action, "--pid", str(pid), *extra] if action == "--get-state" else \
        [CHECKPOINT, "--action", action, "--pid", str(pid), *extra]
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"cuda-checkpoint {action} falhou: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def http(url, method="GET", body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    if method == "POST" and data is None:
        data = b""
    request = urllib.request.Request(url, method=method, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def free_vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, check=True).stdout
    return int(out.split()[0])


class Engine:
    """The real vLLM + cuda-checkpoint side. Tests replace it with a fake."""

    def pids(self):
        return engine_pids()

    def cuda_state(self, pid):
        return checkpoint("--get-state", pid)

    def checkpoint(self, action, pid, *extra):
        checkpoint(action, pid, *extra)

    def vllm(self, path, method="GET", timeout=10):
        return http(VLLM + path, method, timeout=timeout)

    def free_mib(self):
        return free_vram_mib()

    def comfyui(self, path, body=None):
        return http(COMFYUI + path, "GET" if body is None else "POST", body, timeout=10)


class Controller:
    def __init__(self, engine, now=time.monotonic, sleep=time.sleep):
        self.engine, self.now, self.pause = engine, now, sleep
        self.cond = threading.Condition()
        self.leases = {}       # id -> {owner, expires}
        self.operation = None  # 'suspending' | 'resuming' while one runs
        self.sleepers = 0      # pending forced sleeps: new acquires wait for them

    # --- observation -------------------------------------------------------------------
    def _prune(self):
        now = self.now()
        for key in [key for key, lease in self.leases.items() if lease["expires"] <= now]:
            print(f"[gpu-control] lease {key} ({self.leases[key]['owner']}) expired", flush=True)
            del self.leases[key]

    def engine_state(self):
        pids = self.engine.pids()
        if not pids:
            return {"state": "unavailable", "error": "EngineCore do vLLM não encontrado."}
        states = {self.engine.cuda_state(pid) for pid in pids}
        if states == {"checkpointed"}:
            return {"state": "suspended"}
        if states != {"running"}:
            return {"state": "locked", "error": f"Estado CUDA inesperado: {', '.join(sorted(states))}."}
        try:
            self.engine.vllm("/health", timeout=5)
            return {"state": "sleeping" if self.engine.vllm("/is_sleeping", timeout=5)["is_sleeping"] else "awake"}
        except Exception as error:
            return {"state": "unavailable", "error": f"vLLM indisponível: {error}"}

    def state(self):
        with self.cond:
            self._prune()
            operation = self.operation
            leases = [{"owner": lease["owner"]} for lease in self.leases.values()]
        body = {"state": operation} if operation else self.engine_state()
        return {**body, "leases": leases}

    # --- exclusive engine operations -------------------------------------------------------
    def _exclusive(self, name, steps, wait=SLEEP_WAIT):
        deadline = self.now() + wait
        with self.cond:
            while self.operation:
                if self.now() >= deadline:
                    raise Busy("Outra operação de memória está em andamento.")
                self.cond.wait(1)
            self.operation = name
        try:
            steps()
        finally:
            with self.cond:
                self.operation = None
                self.cond.notify_all()

    def _suspend(self):
        pids = self.engine.pids()
        if not pids:
            raise RuntimeError("EngineCore do vLLM não encontrado.")
        if {self.engine.cuda_state(pid) for pid in pids} == {"checkpointed"}:
            return
        if not self.engine.vllm("/is_sleeping")["is_sleeping"]:
            # mode=wait lets in-flight requests from other clients finish first.
            self.engine.vllm("/sleep?level=1&mode=wait", "POST", timeout=600)
        try:
            for pid in pids:
                if self.engine.cuda_state(pid) == "running":
                    self.engine.checkpoint("lock", pid, "--timeout", "10000")
            for pid in pids:
                if self.engine.cuda_state(pid) == "locked":
                    self.engine.checkpoint("checkpoint", pid)
        except Exception:
            self._resume(check_vram=False)  # Never leave the engine locked or half-checkpointed.
            raise

    def _ensure_vram(self, free_comfyui):
        if self.engine.free_mib() >= WAKE_FREE_MIB:
            return
        if free_comfyui:
            queue = self.engine.comfyui("/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                raise Busy("O ComfyUI está gerando e ocupa a GPU. Aguarde a fila terminar.")
            self.engine.comfyui("/free", {"unload_models": True, "free_memory": True})
        deadline = self.now() + VRAM_WAIT
        while True:
            free = self.engine.free_mib()
            if free >= WAKE_FREE_MIB:
                return
            if self.now() >= deadline:
                raise Busy(f"GPU sem VRAM livre para acordar o vLLM: {free} MiB livres, {WAKE_FREE_MIB} necessários.")
            self.pause(0.5)

    def _resume(self, check_vram=True, free_comfyui=True):
        pids = self.engine.pids()
        if not pids:
            raise RuntimeError("EngineCore do vLLM não encontrado.")
        states = {pid: self.engine.cuda_state(pid) for pid in pids}
        sleeping = set(states.values()) != {"running"} or self.engine.vllm("/is_sleeping")["is_sleeping"]
        if not sleeping:
            return False
        if check_vram and set(states.values()) == {"checkpointed"}:
            self._ensure_vram(free_comfyui)
        for pid in pids:
            if self.engine.cuda_state(pid) == "checkpointed":
                self.engine.checkpoint("restore", pid)
        for pid in pids:
            if self.engine.cuda_state(pid) == "locked":
                self.engine.checkpoint("unlock", pid)
        if self.engine.vllm("/is_sleeping")["is_sleeping"]:
            self.engine.vllm("/wake_up", "POST", timeout=300)
        return True

    # --- API -------------------------------------------------------------------------------
    def acquire(self, owner, ttl=DEFAULT_TTL, free_comfyui=True, wait=SLEEP_WAIT):
        owner = str(owner or "desconhecido")[:64]
        ttl = max(30, min(float(ttl), MAX_TTL))
        deadline = self.now() + wait
        with self.cond:
            while self.sleepers:  # a Composer is putting the model to sleep: wait, then wake it again
                if self.now() >= deadline:
                    raise Busy("O modelo está sendo suspenso por outro cliente. Tente novamente.")
                self.cond.wait(1)
            key = uuid.uuid4().hex
            self.leases[key] = {"owner": owner, "expires": self.now() + ttl}
        woke = []
        try:
            self._exclusive("resuming", lambda: woke.append(self._resume(free_comfyui=free_comfyui)),
                            max(1.0, deadline - self.now()))
        except Exception:
            with self.cond:
                self.leases.pop(key, None)
                self.cond.notify_all()
            raise
        print(f"[gpu-control] lease {key} acquired by {owner} (woke={woke[0]})", flush=True)
        return {**self.state(), "lease": key, "woke": woke[0]}

    def release(self, key, sleep=False):
        with self.cond:
            lease = self.leases.pop(str(key), None)
            self.cond.notify_all()
            if not sleep:
                return self.state()
            self.sleepers += 1
        try:
            deadline = self.now() + SLEEP_WAIT
            with self.cond:
                while True:
                    self._prune()
                    if not self.leases:
                        break
                    if self.now() >= deadline:
                        owners = ", ".join(sorted({lease["owner"] for lease in self.leases.values()}))
                        print(f"[gpu-control] forced sleep after waiting for: {owners}", flush=True)
                        break
                    self.cond.wait(1)
            self._exclusive("suspending", self._suspend)
        finally:
            with self.cond:
                self.sleepers -= 1
                self.cond.notify_all()
        print(f"[gpu-control] lease {key} ({lease and lease['owner']}) released with sleep", flush=True)
        return self.state()

    def suspend(self):
        with self.cond:
            self._prune()
            if self.leases:
                owners = ", ".join(sorted({lease["owner"] for lease in self.leases.values()}))
                raise Busy(f"O modelo está em uso por: {owners}.")
            self.sleepers += 1
        try:
            self._exclusive("suspending", self._suspend, wait=5)
        finally:
            with self.cond:
                self.sleepers -= 1
                self.cond.notify_all()
        return self.state()

    def resume(self, free_comfyui=True):
        self._exclusive("resuming", lambda: self._resume(free_comfyui=free_comfyui), wait=5)
        return self.state()


def make_handler(controller):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            data = json.loads(self.rfile.read(min(length, 65536)) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("JSON deve ser um objeto.")
            return data

        def do_GET(self):
            if self.path != "/state":
                return self.reply(404, {"error": "Não encontrado."})
            try:
                self.reply(200, controller.state())
            except Exception as error:
                self.reply(500, {"state": "unavailable", "error": str(error)})

        def do_POST(self):
            try:
                body = self.body()
                action = {
                    "/acquire": lambda: controller.acquire(body.get("owner"), body.get("ttl", DEFAULT_TTL),
                                                           body.get("free_comfyui", True) is not False),
                    "/release": lambda: controller.release(body.get("lease"), body.get("sleep") is True),
                    "/suspend": controller.suspend,
                    "/resume": lambda: controller.resume(body.get("free_comfyui", True) is not False),
                }.get(self.path)
                if not action:
                    return self.reply(404, {"error": "Não encontrado."})
                self.reply(200, action())
            except Busy as error:
                self.reply(409, {"error": str(error), "retry": True})
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except Exception as error:
                print(f"[gpu-control] {self.path} failed: {error}", flush=True)
                try:
                    current = controller.state()
                except Exception:
                    current = {"state": "unavailable"}
                self.reply(500, {**current, "error": str(error)})

        def log_message(self, fmt, *args):
            if not self.path.startswith("/state"):
                print("[gpu-control] " + fmt % args, flush=True)

    return Handler


if __name__ == "__main__":
    port = int(os.environ.get("GPU_CONTROL_PORT", "8001"))
    print(f"[gpu-control] listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(Controller(Engine()))).serve_forever()
