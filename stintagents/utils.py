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


def _create_realtime_runner(agent, voice: str, speed: float, context: dict):
    """Create a RealtimeRunner with standard configuration."""
    from agents.realtime import RealtimeRunner
    
    return RealtimeRunner(
        starting_agent=agent,
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


async def _create_new_session(session_key: str, realtime_agent, conversation_id: str):
    """Create and initialize a new Realtime session."""
    print(f"[INFO] Creating new persistent session for conversation {conversation_id}")
    
    agent_persona = config.AGENT_PERSONAS.get(realtime_agent.name, {})
    voice = agent_persona.get("voice", "alloy")
    speed = agent_persona.get("speed", 1.0)
    
    runner = _create_realtime_runner(realtime_agent, voice, speed, {"conversation_id": conversation_id})
    session = await runner.run(context={"conversation_id": conversation_id})
    await session.__aenter__()
    
    with _SESSION_LOCK:
        _REALTIME_SESSIONS[session_key] = session
    
    return session


async def _close_session(session, session_key: str):
    """Close a session and remove from storage."""
    try:
        await session.close()
        await asyncio.sleep(0.5)
    except Exception as close_err:
        print(f"[WARN] Error during session close: {close_err}")
        await asyncio.sleep(0.5)
    
    with _SESSION_LOCK:
        if session_key in _REALTIME_SESSIONS:
            del _REALTIME_SESSIONS[session_key]


async def _wait_for_pending_transcript(session, user_transcript: str, timeout: float = 2.0):
    """Wait for pending transcript events after a handoff."""
    pending_transcript = None
    wait_start = asyncio.get_event_loop().time()
    
    async for pending_event in session:
        elapsed = asyncio.get_event_loop().time() - wait_start
        if elapsed > timeout:
            print(f"[DEBUG] Timeout waiting for transcript after {elapsed:.2f}s")
            break
        
        print(f"[DEBUG] Pending event type: {pending_event.type}")
        
        if pending_event.type == "history_updated":
            transcript = _extract_user_transcript_from_history(pending_event.history, user_transcript)
            if transcript:
                pending_transcript = transcript
                print(f"[TRANSCRIPT] User said (pending): {pending_transcript}")
                break
        
        elif pending_event.type == "history_added":
            transcript = _extract_transcript_from_item(pending_event.item, "user", user_transcript)
            if transcript:
                pending_transcript = transcript
                print(f"[TRANSCRIPT] User said (added): {pending_transcript}")
                break
        
        elif pending_event.type == "raw":
            raw_data = pending_event.data if hasattr(pending_event, 'data') else None
            if raw_data:
                raw_type = getattr(raw_data, 'type', None)
                print(f"[DEBUG] Raw event type: {raw_type}")
                if raw_type == "conversation.item.input_audio_transcription.completed":
                    transcript = getattr(raw_data, 'transcript', None)
                    if transcript:
                        print(f"[DEBUG] Raw transcript: '{transcript}'")
                        pending_transcript = transcript
                        break
        
        elif pending_event.type in ["audio_end", "error"]:
            print(f"[DEBUG] Breaking on {pending_event.type}")
            break
    
    return pending_transcript


def _extract_user_transcript_from_history(history, existing_transcript: str):
    """Extract user transcript from history items."""
    for item in reversed(history):
        if hasattr(item, 'role') and item.role == 'user':
            transcript = _extract_transcript_from_item(item, "user", existing_transcript)
            if transcript:
                return transcript
    return None


def _extract_transcript_from_item(item, role: str, existing_transcript: str = ""):
    """Extract transcript from a history item."""
    if not (hasattr(item, 'role') and item.role == role):
        return None
    
    if not hasattr(item, 'content'):
        return None
    
    contents = item.content if isinstance(item.content, list) else [item.content]
    for content in contents:
        if role == 'user':
            transcript = getattr(content, 'transcript', None) or getattr(content, 'text', None)
        else:
            transcript = getattr(content, 'text', None) or getattr(content, 'transcript', None)
        
        if transcript and transcript != existing_transcript:
            return transcript
    
    return None


async def _handle_handoff(event, session, session_key: str, conversation_id: str,
                          user_transcript: str, timeout_seconds: int):
    """Handle agent handoff - close old session and create new one with appropriate voice."""
    from_agent = event.from_agent.name
    to_agent_obj = event.to_agent
    to_agent = to_agent_obj.name
    
    print(f"[INFO] Handoff initiated: {from_agent} → {to_agent}, waiting for pending transcripts...")
    
    # Wait for pending transcript
    pending_transcript = await _wait_for_pending_transcript(session, user_transcript)
    last_user_request = pending_transcript or user_transcript or ""
    
    print(f"[CONTEXT] Handoff context: '{last_user_request}'")
    print(f"[INFO] Handoff: {from_agent} → {to_agent}")
    
    # Close current session
    print(f"[INFO] Closing session to switch voice for {to_agent}...")
    await _close_session(session, session_key)
    
    # Get new agent's voice settings
    new_persona = config.AGENT_PERSONAS.get(to_agent, {})
    new_voice = new_persona.get("voice", "alloy")
    new_speed = new_persona.get("speed", 1.0)
    
    print(f"[INFO] Starting new session with voice '{new_voice}' for {to_agent}")
    
    # Build handoff context instruction
    handoff_context_instruction = ""
    if last_user_request:
        handoff_context_instruction = f"""

        # ACTIVE HANDOFF CONTEXT
        You just received a handoff from {from_agent}. The user's most recent request was: "{last_user_request}"
        DO NOT introduce yourself. Respond DIRECTLY to this request using your tools immediately."""
    
    # Clone agent with context
    agent_with_context = to_agent_obj.clone(
        instructions=(to_agent_obj.instructions or "") + handoff_context_instruction
    )
    
    # Create new session
    runner = _create_realtime_runner(agent_with_context, new_voice, new_speed, {"conversation_id": conversation_id})
    new_session = await runner.run(context={"conversation_id": conversation_id})
    await new_session.__aenter__()
    
    with _SESSION_LOCK:
        _REALTIME_SESSIONS[session_key] = new_session
    
    print(f"[INFO] New session active for {to_agent}")
    
    # Trigger response
    trigger_message = last_user_request or "Please help the user with their request."
    print(f"[INFO] Triggering {to_agent} to respond to: '{trigger_message[:50]}...'")
    try:
        await new_session.send_message(trigger_message)
    except Exception as trigger_err:
        print(f"[WARN] Text trigger failed: {trigger_err}")
    
    return new_session, to_agent, last_user_request


async def _collect_response_from_session(session, active_agent: str, timeout_seconds: int,
                                          last_seen_history_len: int = 0):
    """Collect audio and transcript response from a session."""
    response_audio_chunks = []
    response_text = ""
    user_transcript = ""
    last_user_request = ""
    start_time = asyncio.get_event_loop().time()
    
    async for event in session:
        if asyncio.get_event_loop().time() - start_time > timeout_seconds:
            print("[WARN] Response timeout")
            break
        
        event_type = event.type
        
        if event_type == "audio":
            response_audio_chunks.append(event.audio.data)
        
        elif event_type == "agent_start":
            active_agent = event.agent.name
            print(f"[INFO] Agent started: {active_agent}")
        
        elif event_type == "tool_start":
            tool_name = event.tool.name if hasattr(event, 'tool') else 'unknown'
            tool_args = event.arguments if hasattr(event, 'arguments') else '{}'
            print(f"[TOOL] {active_agent} calling: {tool_name} with args: {tool_args}")
            start_time = asyncio.get_event_loop().time()
        
        elif event_type == "tool_end":
            tool_name = event.tool.name if hasattr(event, 'tool') else 'unknown'
            tool_output = event.output if hasattr(event, 'output') else 'no output'
            print(f"[TOOL] {tool_name} result: {str(tool_output)[:200]}")
            start_time = asyncio.get_event_loop().time()
        
        elif event_type == "history_updated":
            history = event.history
            new_items = history[last_seen_history_len:]
            last_seen_history_len = len(history)
            
            for item in new_items:
                if hasattr(item, 'role') and hasattr(item, 'content'):
                    contents = item.content if isinstance(item.content, list) else [item.content]
                    for content in contents:
                        if item.role == 'user':
                            transcript = getattr(content, 'transcript', None) or getattr(content, 'text', None)
                            if transcript:
                                user_transcript = transcript
                                last_user_request = transcript
                                print(f"[TRANSCRIPT] User said: {user_transcript}")
                        elif item.role == 'assistant':
                            transcript = getattr(content, 'text', None) or getattr(content, 'transcript', None)
                            if transcript:
                                response_text = transcript
                                print(f"[TRANSCRIPT] {active_agent} said: {response_text}")
        
        elif event_type == "audio_end":
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
        
        elif event_type == "handoff":
            # Return handoff event for caller to handle
            return {
                "type": "handoff",
                "event": event,
                "audio_chunks": response_audio_chunks,
                "response_text": response_text,
                "user_transcript": user_transcript,
                "last_user_request": last_user_request,
                "active_agent": active_agent,
                "last_seen_history_len": last_seen_history_len
            }
    
    return {
        "type": "complete",
        "audio_chunks": response_audio_chunks,
        "response_text": response_text,
        "user_transcript": user_transcript,
        "last_user_request": last_user_request,
        "active_agent": active_agent,
        "last_seen_history_len": last_seen_history_len
    }


async def _save_response_to_session(conversation_id: str, response_text: str, active_agent: str):
    """Save response transcript to session for evaluation."""
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
            # Extract and process audio
            sample_rate, raw_audio = audio_data if isinstance(audio_data, tuple) else (24000, audio_data)
            
            if hasattr(raw_audio, 'size') and raw_audio.size == 0:
                return None, None
            
            pcm16_audio, _ = preprocess_audio_for_realtime(raw_audio, sample_rate)
            audio_bytes = pcm16_audio.tobytes()
            
            # Get or create session
            session_key = conversation_id
            with _SESSION_LOCK:
                if session_key not in _REALTIME_SESSIONS:
                    session = await _create_new_session(session_key, realtime_agent, conversation_id)
                else:
                    session = _REALTIME_SESSIONS[session_key]
            
            print(f"[INFO] Processing {len(pcm16_audio)} samples with session {session_key}")
            
            # Send audio input
            if len(audio_bytes) > 0:
                await session.send_audio(audio_bytes, commit=False)
            
            # Collect response with handoff handling
            active_agent = realtime_agent.name
            timeout_seconds = 15
            
            result = await _collect_response_from_session(session, active_agent, timeout_seconds)
            
            # Handle handoff if needed
            while result["type"] == "handoff":
                session, active_agent, _ = await _handle_handoff(
                    result["event"], session, session_key, conversation_id,
                    result["user_transcript"], timeout_seconds
                )
                
                # Collect response from new agent
                result = await _collect_response_from_session(session, active_agent, timeout_seconds)
            
            # Build output audio
            response_audio_chunks = result["audio_chunks"]
            response_text = result["response_text"]
            active_agent = result["active_agent"]
            
            if response_audio_chunks:
                combined_audio = b"".join(response_audio_chunks)
                audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                output_audio = (24000, audio_array)
                
                if response_text:
                    await _save_response_to_session(conversation_id, response_text, active_agent)
                
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
