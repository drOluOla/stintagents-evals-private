"""
StintAgents Voice AI Package
Multi-agent voice interaction system for employee onboarding
"""

__version__ = "0.1.0"

from .utils import (
    get_or_create_event_loop,
    get_or_create_session,
    preprocess_audio,
    convert_audio_bytes,
    generate_speech_async,
    transcribe_audio_async,
    transcribe_audio_batch_async,
    get_agent_response_with_speech,
    process_voice_input,
    process_audio_batch,
    process_audio_batch_async,
    # Multi-GPU
    enable_multi_gpu_transcription,
    get_multi_gpu_pool,
    MultiGPUTranscriptionPool,
    GPUWorkerConfig,
)

from .ui import create_agent_avatar, create_gradio_interface

__all__ = [
    # Utils
    "get_or_create_event_loop",
    "get_or_create_session",
    "preprocess_audio",
    "convert_audio_bytes",
    "generate_speech_async",
    "transcribe_audio_async",
    "transcribe_audio_batch_async",
    "get_agent_response_with_speech",
    "process_voice_input",
    "process_audio_batch",
    "process_audio_batch_async",
    # Multi-GPU
    "enable_multi_gpu_transcription",
    "get_multi_gpu_pool",
    "MultiGPUTranscriptionPool",
    "GPUWorkerConfig",
    # UI
    "create_agent_avatar",
    "create_gradio_interface",
]
