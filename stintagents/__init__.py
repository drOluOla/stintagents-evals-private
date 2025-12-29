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
    get_agent_response_with_speech,
    process_voice_input,
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
