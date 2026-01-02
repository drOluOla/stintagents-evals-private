"""
StintAgents Voice AI Package
Multi-agent voice interaction system for employee onboarding
"""

__version__ = "0.2.0"

from .utils import (
    get_or_create_event_loop,
    get_or_create_session,
    preprocess_audio_for_realtime,
    audio_to_base64_pcm16,
    base64_pcm16_to_audio,
    process_voice_input_realtime,
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
    "get_agent_response_with_speech",
    "process_voice_input",
    # UI
    "create_agent_avatar",
    "create_gradio_interface",
]
