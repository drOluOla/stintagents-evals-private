"""
Core utilities for StintAgents Voice AI
Audio processing, transcription, TTS, and agent coordination
"""
import asyncio
import threading
import numpy as np
import torch
import io
import json
from typing import Optional, Tuple
from scipy import signal
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from pydub import AudioSegment

import stintagents.config as config

# ==============================================================================
# EVENT LOOP
# ==============================================================================
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()

def get_or_create_event_loop():
    global _event_loop
    with _loop_lock:
        if _event_loop is None or not _event_loop.is_running():
            _event_loop = asyncio.new_event_loop()
            threading.Thread(
                target=lambda: (asyncio.set_event_loop(_event_loop), _event_loop.run_forever()),
                daemon=True
            ).start()
    return _event_loop

# ==============================================================================
# WHISPER INITIALIZATION
# ==============================================================================
USE_GPU = torch.cuda.is_available()
device = "cuda" if USE_GPU else "cpu"
model_size = "base" if USE_GPU else "tiny"
compute_type = "float16" if USE_GPU else "int8"

print(f"[init] Using {device.upper()}{f': {torch.cuda.get_device_name(0)}' if USE_GPU else ''}")
print(f"[init] Loading Whisper model from HuggingFace (first run may take a moment)...")

WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)

def get_or_create_session(conversation_id: str):
    """Get or create a session - SQLiteSession should be imported in notebook"""
    # Note: SQLiteSession import happens in the notebook
    # This function expects CONVERSATION_SESSIONS to be populated
    if conversation_id not in config.CONVERSATION_SESSIONS:
        # Will be handled by notebook initialization
        raise ValueError(f"Session {conversation_id} not found. Initialize sessions in notebook first.")
    return config.CONVERSATION_SESSIONS[conversation_id]

async_openai_client = AsyncOpenAI()

# ==============================================================================
# AUDIO PROCESSING
# ==============================================================================
def preprocess_audio(raw_audio, sample_rate):
    """Normalize, resample, and convert audio to mono float32 @ 16kHz."""
    if not isinstance(raw_audio, np.ndarray):
        raw_audio = np.array(raw_audio)
    
    if raw_audio.size == 0:
        return np.array([], dtype=np.float32), 16000
    
    # Convert to float32
    if raw_audio.dtype in (np.int16, np.int32):
        raw_audio = raw_audio.astype(np.float32) / (32768.0 if raw_audio.dtype == np.int16 else 2147483648.0)
    
    # Mono conversion
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1, dtype=np.float32)
    
    # Normalize
    max_amp = np.abs(raw_audio).max()
    if max_amp > 0:
        raw_audio *= 0.95 / max_amp
    
    # Resample to 16kHz
    if sample_rate != 16000:
        raw_audio = signal.resample(raw_audio, int(len(raw_audio) * 16000 / sample_rate)).astype(np.float32)
    
    return raw_audio, 16000

def convert_audio_bytes(audio_bytes: bytes, format: str = "mp3"):
    """Convert TTS audio bytes to (sample_rate, numpy_array) for Gradio."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1).astype(np.int16)
        
        return (audio.frame_rate, samples)
    except Exception as e:
        print(f"[ERROR] Audio conversion: {e}")
        return None

# ==============================================================================
# TTS & TRANSCRIPTION
# ==============================================================================
async def generate_speech_async(text: str, agent_name: str = "HR Manager") -> Optional[bytes]:
    """Generate speech using OpenAI TTS."""
    try:
        personas = config.AGENT_PERSONAS
        agent_cfg = personas.get(agent_name, personas.get("HR Manager", {}))
        response = await async_openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=agent_cfg.get("voice", "alloy"),
            input=text,
            speed=agent_cfg.get("speed", 1.0),
            response_format="mp3"
        )
        return response.content
    except Exception as e:
        print(f"[ERROR] TTS: {e}")
        return None

async def transcribe_audio_async(audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
    """Transcribe audio using faster-whisper."""
    try:
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            segments, _ = WHISPER_MODEL.transcribe(
                audio, beam_size=1, language="en", vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, threshold=0.5),
                temperature=0.0, no_speech_threshold=0.6
            )
            return " ".join(seg.text for seg in segments).strip()
        
        return await loop.run_in_executor(None, _transcribe)
    except Exception as e:
        print(f"[ERROR] Transcription: {e}")
        return None

# ==============================================================================
# AGENT RESPONSE
# ==============================================================================
async def get_agent_response_with_speech(user_input: str, conversation_id: str = "default", runner=None, hr_manager_agent=None):
    """
    Get agent response and TTS in one call.
    
    Args:
        user_input: User's transcribed text
        conversation_id: Session identifier
        runner: Runner instance (from openai_agent.runner import Runner)
        hr_manager_agent: The hr_manager agent instance
    """
    try:
        if runner is None or hr_manager_agent is None:
            raise ValueError("Runner and hr_manager_agent must be provided")
            
        config.CURRENT_TOOL_EXPECTED.clear()
        session = get_or_create_session(conversation_id)
        
        agent_result = await runner.run(hr_manager_agent, input=user_input, session=session)
        agent_name = agent_result.last_agent.name if agent_result.last_agent else "HR Manager"
        response = agent_result.final_output
        
        # Save metadata
        await session.add_items([{
            "role": "system",
            "content": json.dumps({
                "evaluation_metadata": True,
                "responding_agent": agent_name,
                "expected_response": config.CURRENT_TOOL_EXPECTED.get("expected", response),
            })
        }])
        config.CURRENT_TOOL_EXPECTED.clear()
        
        speech_bytes = await generate_speech_async(response, agent_name)
        return response, agent_name, speech_bytes
    
    except Exception as e:
        print(f"[ERROR] Agent response: {e}")
        return ("I apologize, but I'm having trouble processing your request.", "HR Manager", None)

# ==============================================================================
# VOICE PIPELINE
# ==============================================================================
def process_voice_input(audio_data, conversation_id: str = "default", runner=None, hr_manager_agent=None):
    """
    Complete voice processing pipeline - returns (audio, agent_name).
    
    Args:
        audio_data: Audio input from Gradio
        conversation_id: Session identifier
        runner: Runner instance (from openai_agent.runner import Runner)
        hr_manager_agent: The hr_manager agent instance
    """
    
    if audio_data is None:
        return None, None
    
    async def _process():
        try:
            # Extract audio
            sample_rate, raw_audio = audio_data if isinstance(audio_data, tuple) else (24000, audio_data)
            
            if hasattr(raw_audio, 'size') and raw_audio.size == 0:
                return None, None
            
            # Process pipeline
            processed_audio, _ = preprocess_audio(raw_audio, sample_rate)
            transcription = await transcribe_audio_async(processed_audio)
            
            if not transcription:
                return None, None
            
            response_text, active_agent, speech_bytes = await get_agent_response_with_speech(
                transcription, conversation_id, runner, hr_manager_agent
            )
            
            # Convert speech to playable format
            output_audio = convert_audio_bytes(speech_bytes, "mp3") if speech_bytes else None
            
            return output_audio, active_agent
        
        except Exception as e:
            print(f"[ERROR] Pipeline: {e}")
            return None, None
    
    # Run async code in background event loop
    loop = get_or_create_event_loop()
    future = asyncio.run_coroutine_threadsafe(_process(), loop)
    
    try:
        return future.result(timeout=30)
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None

# Initialize
get_or_create_event_loop()
print("[init] Ready!")
