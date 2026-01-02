"""
Core utilities for StintAgents Voice AI
Real-time audio processing using OpenAI Realtime API with official SDK
"""
import asyncio
import threading
import numpy as np
import io
import json
import base64
from typing import Optional, Tuple
from scipy import signal

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
# SESSION MANAGEMENT
# ==============================================================================
def get_or_create_session(conversation_id: str):
    """Get or create a session - SQLiteSession should be imported in notebook"""
    if conversation_id not in config.CONVERSATION_SESSIONS:
        raise ValueError(f"Session {conversation_id} not found. Initialize sessions in notebook first.")
    return config.CONVERSATION_SESSIONS[conversation_id]

# ==============================================================================
# AUDIO PROCESSING FOR REALTIME API
# ==============================================================================
def preprocess_audio_for_realtime(raw_audio, sample_rate):
    """Convert audio to PCM16 mono @ 24kHz for Realtime API."""
    if not isinstance(raw_audio, np.ndarray):
        raw_audio = np.array(raw_audio)
    
    if raw_audio.size == 0:
        return np.array([], dtype=np.int16), 24000
    
    # Convert to float32 first
    if raw_audio.dtype in (np.int16, np.int32):
        raw_audio = raw_audio.astype(np.float32) / (32768.0 if raw_audio.dtype == np.int16 else 2147483648.0)
    
    # Mono conversion
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1, dtype=np.float32)
    
    # Normalize
    max_amp = np.abs(raw_audio).max()
    if max_amp > 0:
        raw_audio *= 0.95 / max_amp
    
    # Resample to 24kHz (Realtime API requirement)
    if sample_rate != 24000:
        raw_audio = signal.resample(raw_audio, int(len(raw_audio) * 24000 / sample_rate)).astype(np.float32)
    
    # Convert to PCM16
    pcm16_audio = (raw_audio * 32767.0).astype(np.int16)
    
    return pcm16_audio, 24000

def audio_to_base64_pcm16(audio_array: np.ndarray) -> str:
    """Convert numpy array to base64-encoded PCM16 string."""
    return base64.b64encode(audio_array.tobytes()).decode('utf-8')

def base64_pcm16_to_audio(base64_str: str, sample_rate: int = 24000) -> Tuple[int, np.ndarray]:
    """Convert base64-encoded PCM16 to (sample_rate, numpy_array) for Gradio."""
    audio_bytes = base64.b64decode(base64_str)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    return (sample_rate, audio_array)


# ==============================================================================
# REALTIME VOICE PIPELINE (SDK-based)
# ==============================================================================
def process_voice_input_realtime(audio_data, conversation_id: str = "default", realtime_agent=None):
    """
    Process voice input using RealtimeAgent from official SDK - returns (audio, agent_name).
    
    Args:
        audio_data: Audio input from Gradio
        conversation_id: Session identifier
        realtime_agent: RealtimeAgent instance from agents.realtime
    """
    
    if audio_data is None or realtime_agent is None:
        return None, None
    
    async def _process():
        try:
            from agents.realtime import RealtimeRunner
            import os
            
            # Extract and process audio
            sample_rate, raw_audio = audio_data if isinstance(audio_data, tuple) else (24000, audio_data)
            
            if hasattr(raw_audio, 'size') and raw_audio.size == 0:
                return None, None
            
            # Convert to PCM16 @ 24kHz for Realtime API
            pcm16_audio, _ = preprocess_audio_for_realtime(raw_audio, sample_rate)
            
            # Convert to raw bytes for send_audio
            audio_bytes = pcm16_audio.tobytes()
            
            print(f"[INFO] Processing {len(pcm16_audio)} samples with {realtime_agent.name}")
            
            # Create RealtimeRunner and start session
            runner = RealtimeRunner(
                starting_agent=realtime_agent,
                config={
                    "model_settings": {
                        "model_name": "gpt-4o-realtime-preview-2024-12-17",
                        "modalities": ["audio"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                            "create_response": True,
                        }
                    }
                }
            )
            
            # Start session and send audio
            session = await runner.run(context={"conversation_id": conversation_id})
            
            # Collect response audio
            response_audio_chunks = []
            response_text = ""
            active_agent = realtime_agent.name
            
            async with session:
                # Send audio input - don't commit, let turn detection handle it
                if len(audio_bytes) > 0:
                    await session.send_audio(audio_bytes, commit=False)
                
                # Listen for response events
                async for event in session:
                    event_type = event.type
                    
                    if event_type == "audio":
                        # Collect audio chunks
                        response_audio_chunks.append(event.audio.data)
                    
                    elif event_type == "history_updated":
                        # Extract transcript from history
                        for item in event.history:
                            if hasattr(item, 'role') and item.role == 'assistant':
                                if hasattr(item, 'content'):
                                    for content in item.content if isinstance(item.content, list) else [item.content]:
                                        if hasattr(content, 'text'):
                                            response_text += content.text
                    
                    elif event_type == "audio_end":
                        # Response complete
                        print(f"[INFO] Response complete: {response_text[:100] if response_text else 'no transcript'}...")
                        break
                    
                    elif event_type == "error":
                        print(f"[ERROR] Realtime API error: {event.error if hasattr(event, 'error') else 'unknown'}")
                        break
            
            # Combine audio chunks and convert to Gradio format
            if response_audio_chunks:
                combined_audio = b"".join(response_audio_chunks)
                audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                output_audio = (24000, audio_array)
                
                # Save transcript to session for evaluation
                if response_text:
                    session_obj = get_or_create_session(conversation_id)
                    await session_obj.add_items([
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "system",
                            "content": json.dumps({
                                "evaluation_metadata": True,
                                "responding_agent": active_agent,
                                "expected_response": config.CURRENT_TOOL_EXPECTED.get("expected", response_text),
                            })
                        }
                    ])
                    config.CURRENT_TOOL_EXPECTED.clear()
                
                return output_audio, active_agent
            
            return None, None
        
        except Exception as e:
            print(f"[ERROR] Realtime pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    # Run async code in background event loop
    loop = get_or_create_event_loop()
    future = asyncio.run_coroutine_threadsafe(_process(), loop)
    
    try:
        return future.result(timeout=30)
    except Exception as e:
        print(f"[ERROR] Future execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Initialize
get_or_create_event_loop()
print("[init] Realtime API ready!")
