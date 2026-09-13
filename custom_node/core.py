"""Pure payload construction shared by the node and its tests."""
import base64
import io
import re
from PIL import Image


def selected_slot(value):
    match = re.match(r'^system_prompt_([1-9]|1[0-6])(?:\s*::.*)?$', str(value))
    if not match:
        raise ValueError('Select a connected system prompt.')
    return 'system_prompt_' + match.group(1)


def build_messages(prompt, active_prompt, inputs, video_fps=24.0,
                   video_sample_fps=1.0, max_video_frames=32, max_image_side=1536):
    slot = selected_slot(active_prompt)
    system = inputs.get(slot)
    if not isinstance(system, str) or not system.strip():
        raise ValueError(f'{slot} is disconnected or empty. Select a connected system prompt.')
    if not prompt.strip():
        raise ValueError('Enter a request to compose.')
    content = [{'type': 'text', 'text': prompt}]
    image_count = 0
    video_count = 0

    def encode(frame):
        array = (frame.detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype('uint8')
        image = Image.fromarray(array, mode='RGB')
        image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()

    def validate(batch, name):
        if batch.ndim != 4 or batch.shape[-1] != 3 or min(batch.shape) < 1:
            raise ValueError(f'{name} must be a nonempty IMAGE batch [B,H,W,3].')

    for number in range(1, 9):
        batch = inputs.get(f'image_{number}')
        if batch is None:
            continue
        validate(batch, f'image_{number}')
        for index, frame in enumerate(batch):
            image_count += 1
            if image_count > 32:
                raise ValueError('At most 32 reference images are supported. No images were silently dropped.')
            label = f'<Picture {image_count}>: image_{number}, batch item {index + 1}. Reference role is defined by the user.'
            content.extend([{'type': 'text', 'text': label},
                            {'type': 'image_url', 'image_url': {'url': encode(frame)}}])
    video = inputs.get('video')
    if video is not None:
        validate(video, 'video')
        if video_fps <= 0 or video_sample_fps <= 0:
            raise ValueError('Video frame rates must be positive.')
        step = max(1, round(video_fps / video_sample_fps))
        indices = list(range(0, len(video), step))
        if indices[-1] != len(video) - 1:
            indices.append(len(video) - 1)
        if len(indices) > max_video_frames:
            indices = sorted({round(i * (len(video) - 1) / (max_video_frames - 1)) for i in range(max_video_frames)})
        content.append({'type': 'text', 'text': (
            f'Video reference: {len(video)} input frames at {video_fps:g} FPS; '
            f'{len(indices)} sampled frames follow in chronological order. '
            'These are frames of one video, not separate characters. '
            'Sampling omits intermediate motion; no audio is supplied.')})
        for index in indices:
            video_count += 1
            content.extend([{'type': 'text', 'text': f'<Video 1, frame {video_count}, t={index / video_fps:.3f}s>'},
                            {'type': 'image_url', 'image_url': {'url': encode(video[index])}}])
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}], {
        'system_slot': slot, 'images': image_count, 'video_frames': video_count}
