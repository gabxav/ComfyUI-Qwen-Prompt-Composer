"""python3 -m unittest discover -s tests (from the repository root)."""
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import gpu_control  # noqa: E402


class FakeEngine:
    def __init__(self):
        self.cuda = "running"
        self.sleeping = False
        self.free = 31000
        self.comfy_queue = {"queue_running": [], "queue_pending": []}
        self.calls = []

    def pids(self):
        return [457]

    def cuda_state(self, pid):
        return self.cuda

    def checkpoint(self, action, pid, *extra):
        self.calls.append(action)
        self.cuda = {"lock": "locked", "checkpoint": "checkpointed", "restore": "locked", "unlock": "running"}[action]
        if action == "checkpoint":
            self.free += 1800
        if action == "restore":
            self.free -= 1800

    def vllm(self, path, method="GET", timeout=10):
        self.calls.append(path)
        if path.startswith("/sleep"):
            self.sleeping, self.free = True, self.free + 22000
        if path == "/wake_up":
            self.sleeping, self.free = False, self.free - 22000
        return {"is_sleeping": self.sleeping} if path == "/is_sleeping" else {}

    def free_mib(self):
        return self.free

    def comfyui(self, path, body=None):
        self.calls.append("comfyui" + path)
        if path == "/free":
            self.free += 20000
        return self.comfy_queue


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.controller = gpu_control.Controller(self.engine, sleep=lambda _: None)

    def suspend_engine(self):
        self.controller.suspend()
        self.assertEqual(self.controller.state()["state"], "suspended")
        self.engine.calls.clear()

    def test_acquire_wakes_a_suspended_model_and_release_keeps_it_awake(self):
        self.suspend_engine()
        lease = self.controller.acquire("ai-chat")
        self.assertTrue(lease["woke"])
        self.assertEqual(lease["state"], "awake")
        order = [call for call in self.engine.calls if call in ("restore", "unlock", "/wake_up")]
        self.assertEqual(order, ["restore", "unlock", "/wake_up"])
        self.assertEqual(self.controller.release(lease["lease"])["state"], "awake")
        self.assertEqual(self.controller.state()["leases"], [])

    def test_acquire_on_awake_model_does_not_touch_the_engine(self):
        lease = self.controller.acquire("h3-prompt-enhancer")
        self.assertFalse(lease["woke"])
        self.assertNotIn("restore", self.engine.calls)
        self.assertEqual(lease["leases"], [{"owner": "h3-prompt-enhancer"}])

    def test_manual_suspend_is_refused_while_leased(self):
        lease = self.controller.acquire("comfyui-composer")
        with self.assertRaisesRegex(gpu_control.Busy, "comfyui-composer"):
            self.controller.suspend()
        self.controller.release(lease["lease"])
        self.assertEqual(self.controller.suspend()["state"], "suspended")

    def test_release_with_sleep_waits_for_other_leases_then_suspends(self):
        composer = self.controller.acquire("comfyui-composer")
        chat = self.controller.acquire("ai-chat")
        done = []
        thread = threading.Thread(target=lambda: done.append(self.controller.release(composer["lease"], sleep=True)))
        thread.start()
        time.sleep(0.3)
        self.assertEqual(done, [])  # still waiting for ai-chat
        self.assertEqual(self.engine.cuda, "running")
        self.controller.release(chat["lease"])
        thread.join(5)
        self.assertEqual(done[0]["state"], "suspended")

    def test_acquire_waits_for_a_pending_forced_sleep_then_wakes_again(self):
        composer = self.controller.acquire("comfyui-composer")
        chat = self.controller.acquire("ai-chat")
        threading.Thread(target=lambda: self.controller.release(composer["lease"], sleep=True)).start()
        time.sleep(0.2)
        results = []
        waiter = threading.Thread(target=lambda: results.append(self.controller.acquire("h3-prompt-enhancer")))
        waiter.start()
        time.sleep(0.2)
        self.assertEqual(results, [])  # blocked behind the pending sleep
        self.controller.release(chat["lease"])
        waiter.join(5)
        self.assertTrue(results[0]["woke"])
        self.assertIn("checkpoint", self.engine.calls)
        self.assertEqual(results[0]["state"], "awake")

    def test_short_vram_frees_idle_comfyui_before_waking(self):
        self.suspend_engine()
        self.engine.free = 10000
        lease = self.controller.acquire("ai-chat")
        self.assertIn("comfyui/free", self.engine.calls)
        self.assertEqual(lease["state"], "awake")

    def test_busy_comfyui_refuses_to_wake_and_drops_the_lease(self):
        self.suspend_engine()
        self.engine.free = 10000
        self.engine.comfy_queue = {"queue_running": [["job"]], "queue_pending": []}
        with self.assertRaisesRegex(gpu_control.Busy, "ComfyUI"):
            self.controller.acquire("ai-chat")
        self.assertEqual(self.controller.state()["leases"], [])
        self.assertEqual(self.engine.cuda, "checkpointed")

    def test_composer_skips_comfyui_free_and_waits_for_vram(self):
        self.suspend_engine()
        self.engine.free = 10000
        gpu_control.VRAM_WAIT, previous = 0, gpu_control.VRAM_WAIT
        try:
            with self.assertRaisesRegex(gpu_control.Busy, "VRAM"):
                self.controller.acquire("comfyui-composer", free_comfyui=False)
        finally:
            gpu_control.VRAM_WAIT = previous
        self.assertNotIn("comfyui/free", self.engine.calls)

    def test_expired_leases_do_not_block_suspend(self):
        clock = [0.0]
        controller = gpu_control.Controller(self.engine, now=lambda: clock[0], sleep=lambda _: None)
        controller.acquire("crashed-client", ttl=30)
        clock[0] = 31
        self.assertEqual(controller.suspend()["state"], "suspended")

    def test_failed_checkpoint_restores_the_engine(self):
        original = self.engine.checkpoint

        def failing(action, pid, *extra):
            if action == "checkpoint":
                raise RuntimeError("checkpoint falhou")
            original(action, pid, *extra)
        self.engine.checkpoint = failing
        with self.assertRaisesRegex(RuntimeError, "checkpoint falhou"):
            self.controller.suspend()
        self.assertEqual(self.engine.cuda, "running")
        self.assertFalse(self.engine.sleeping)


if __name__ == "__main__":
    unittest.main()
