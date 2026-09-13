import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('core', Path(__file__).parents[1] / 'custom_node/core.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

class Tensor:
    def __init__(self, data): self.data = np.array(data); self.shape = self.data.shape; self.ndim = self.data.ndim
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return Tensor(self.data[index])
    def detach(self): return self
    def cpu(self): return self
    def float(self): return self
    def clamp(self, lo, hi): return Tensor(np.clip(self.data, lo, hi))
    def numpy(self): return self.data


def test_only_selected_system_prompt_is_sent():
    messages, meta = c.build_messages('pedido', 'system_prompt_2 :: Renamed', {'system_prompt_1': 'UNUSED', 'system_prompt_2': 'EXACT\nTEXT'})
    assert messages[0]['content'] == 'EXACT\nTEXT' and 'UNUSED' not in str(messages)
    with pytest.raises(ValueError): c.build_messages('pedido', 'system_prompt_3', {'system_prompt_1': 'A'})


def test_images_and_video_remain_ordered_and_labeled():
    batch = Tensor(np.zeros((2, 8, 8, 3)))
    video = Tensor(np.zeros((49, 8, 8, 3)))
    messages, meta = c.build_messages('Scene', 'system_prompt_1', {'system_prompt_1': 'S', 'image_3': batch, 'video': video}, max_video_frames=3)
    parts = messages[1]['content']; labels = [p['text'] for p in parts if p['type'] == 'text']
    assert meta == {'system_slot': 'system_prompt_1', 'images': 2, 'video_frames': 3}
    assert '<Picture 1>' in labels[1] and '<Picture 2>' in labels[2]
    assert 't=0.000s' in labels[4] and 't=2.000s' in labels[-1]
    assert len([p for p in parts if p['type'] == 'image_url']) == 5


def test_excess_images_are_not_silently_dropped():
    with pytest.raises(ValueError, match='32'):
        c.build_messages('Scene', 'system_prompt_1', {'system_prompt_1': 'S', 'image_1': Tensor(np.zeros((33, 2, 2, 3)))})
