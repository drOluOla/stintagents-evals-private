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

# Global session storage for persistent Realtime sessions
_REALTIME_SESSIONS = {}
_SESSION_LOCK = threading.Lock()

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
            
            # Get or create persistent session for this conversation
            session_key = conversation_id  # One session per conversation, not per agent
            session = None
            
            with _SESSION_LOCK:
                if session_key not in _REALTIME_SESSIONS:
                    print(f"[INFO] Creating new persistent session for conversation {conversation_id}")
                    
                    # Get voice settings from agent persona
                    agent_persona = config.AGENT_PERSONAS.get(realtime_agent.name, {})
                    voice = agent_persona.get("voice", "alloy")
                    speed = agent_persona.get("speed", 1.0)
                    
                    # Create RealtimeRunner
                    runner = RealtimeRunner(
                        starting_agent=realtime_agent,
                        config={
                            "model_settings": {
                                "model_name": "gpt-4o-realtime-preview-2024-12-17",
                                "modalities": ["audio"],
                                "voice": voice,
                                "speed": speed,
                                "input_audio_format": "pcm16",
                                "output_audio_format": "pcm16",
                                "input_audio_transcription": {
                                    "model": "whisper-1"
                                },
                                "turn_detection": {
                                    "type": "server_vad",
                                    "threshold": 0.5,
                                    "prefix_padding_ms": 300,
                                    "silence_duration_ms": 500,
                                    "create_response": True,
                                },
                            }
                        }
                    )
                    
                    # Start persistent session
                    session = await runner.run(context={"conversation_id": conversation_id})
                    await session.__aenter__()  # Enter the context manager
                    _REALTIME_SESSIONS[session_key] = session
                else:
                    session = _REALTIME_SESSIONS[session_key]
            
            print(f"[INFO] Processing {len(pcm16_audio)} samples with session {session_key}")
            
            # Collect response audio and transcripts
            response_audio_chunks = []
            response_text = ""
            user_transcript = ""
            active_agent = realtime_agent.name
            
            # Send audio input - don't commit, let turn detection handle it
            if len(audio_bytes) > 0:
                await session.send_audio(audio_bytes, commit=False)
            
            # Listen for response events (with timeout to avoid hanging)
            timeout_seconds = 15
            start_time = asyncio.get_event_loop().time()
            
            async for event in session:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                    print("[WARN] Response timeout, returning partial results")
                    break
                
                event_type = event.type
                
                if event_type == "audio":
                    # Collect audio chunks
                    response_audio_chunks.append(event.audio.data)
                
                elif event_type == "handoff":
                    # Track agent handoff - restart session to enable unique voice per agent
                    from_agent = event.from_agent.name
                    to_agent_obj = event.to_agent
                    to_agent = to_agent_obj.name
                    active_agent = to_agent
                    print(f"[INFO] Handoff: {from_agent} → {to_agent}")
                    
                    # Capture conversation history before closing session
                    conversation_history = []
                    if hasattr(session, 'model') and hasattr(session.model, '_conversation_items'):
                        conversation_history = list(session.model._conversation_items)
                    print(f"[INFO] Preserving {len(conversation_history)} conversation items")
                    
                    # Extract last user message for context
                    last_user_message = ""
                    for item in reversed(conversation_history):
                        if hasattr(item, 'role') and item.role == 'user':
                            if hasattr(item, 'content'):
                                contents = item.content if isinstance(item.content, list) else [item.content]
                                for content in contents:
                                    if hasattr(content, 'transcript') and content.transcript:
                                        last_user_message = content.transcript
                                        break
                                    elif hasattr(content, 'text') and content.text:
                                        last_user_message = content.text
                                        break
                            if last_user_message:
                                break
                    
                    print(f"[DEBUG] Last user message: '{last_user_message}'")
                    
                    # Close current session gracefully
                    print(f"[INFO] Closing session to switch voice for {to_agent}...")
                    try:
                        await session.close()
                    except:
                        pass
                    
                    with _SESSION_LOCK:
                        if session_key in _REALTIME_SESSIONS:
                            del _REALTIME_SESSIONS[session_key]
                    
                    # Get new agent's voice settings from persona
                    new_persona = config.AGENT_PERSONAS.get(to_agent, {})
                    new_voice = new_persona.get("voice", "alloy")
                    new_speed = new_persona.get("speed", 1.0)
                    
                    print(f"[INFO] Starting new session with voice '{new_voice}' for {to_agent}")
                    
                    # Create new runner with the target agent
                    from agents.realtime import RealtimeRunner
                    runner = RealtimeRunner(
                        starting_agent=to_agent_obj,
                        config={
                            "model_settings": {
                                "model_name": "gpt-4o-realtime-preview-2024-12-17",
                                "modalities": ["audio", "text"],
                                "voice": new_voice,
                                "speed": new_speed,
                                "input_audio_format": "pcm16",
                                "output_audio_format": "pcm16",
                                "input_audio_transcription": {
                                    "model": "whisper-1"
                                },
                                "turn_detection": {
                                    "type": "server_vad",
                                    "threshold": 0.5,
                                    "prefix_padding_ms": 300,
                                    "silence_duration_ms": 500,
                                    "create_response": True,
                                },
                            }
                        }
                    )
                    
                    # Start new session
                    session = await runner.run(context={"conversation_id": conversation_id})
                    await session.__aenter__()
                    
                    # Store new session
                    with _SESSION_LOCK:
                        _REALTIME_SESSIONS[session_key] = session
                    
                    print(f"[INFO] New session active for {to_agent}")
                    
                    # Trigger immediate response by sending a message
                    # This makes the agent aware of the handoff context and prompts them to speak
                    try:
                        # Create a context message that triggers the agent to respond
                        # The agent's instructions already tell them to respond naturally after handoff
                        handoff_trigger = (
                            f"Please help me with: {last_user_message}" if last_user_message 
                            else "I need your help."
                        )
                        
                        # send_message accepts a string directly
                        await session.send_message(handoff_trigger)
                        print(f"[INFO] Sent message to trigger {to_agent}'s immediate response")
                        
                    except Exception as e:
                        print(f"[WARN] Failed to trigger immediate response: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Reset timeout and continue listening on new session
                    start_time = asyncio.get_event_loop().time()
                    continue
                
                elif event_type == "agent_start":
                    # Track which agent is responding
                    active_agent = event.agent.name
                    print(f"[INFO] Agent started: {active_agent}")
                
                elif event_type == "history_updated":
                    # Extract transcripts from history (both user and assistant)
                    for item in event.history:
                        if hasattr(item, 'role'):
                            if hasattr(item, 'content'):
                                for content in item.content if isinstance(item.content, list) else [item.content]:
                                    # User input transcript
                                    if item.role == 'user':
                                        if hasattr(content, 'transcript') and content.transcript:
                                            user_transcript = content.transcript
                                            print(f"[TRANSCRIPT] User said: {user_transcript}")
                                        elif hasattr(content, 'text') and content.text:
                                            user_transcript = content.text
                                            print(f"[TRANSCRIPT] User said: {user_transcript}")
                                    # Assistant response transcript
                                    elif item.role == 'assistant':
                                        if hasattr(content, 'text') and content.text:
                                            response_text = content.text
                                            print(f"[TRANSCRIPT] {active_agent} said: {response_text}")
                                        elif hasattr(content, 'transcript') and content.transcript:
                                            response_text = content.transcript
                                            print(f"[TRANSCRIPT] {active_agent} said: {response_text}")
                
                elif event_type == "audio_end":
                    # Response complete - show full transcript for debugging
                    if user_transcript:
                        print(f"[SUMMARY] User input: '{user_transcript}'")
                    if response_text:
                        print(f"[SUMMARY] {active_agent} response: '{response_text}'")
                    else:
                        print(f"[WARN] No transcript captured for {active_agent}'s response")
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
