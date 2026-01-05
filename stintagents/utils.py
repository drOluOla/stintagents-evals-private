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
            last_seen_history_len = 0  # Track history length to only process new items
            last_user_request = ""  # Track the user's last request for handoff context
            
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
                
                elif event_type == "tool_start":
                    # SDK executes tools automatically - just log for visibility
                    tool_name = event.tool.name if hasattr(event, 'tool') else 'unknown'
                    tool_args = event.arguments if hasattr(event, 'arguments') else '{}'
                    print(f"[TOOL] {active_agent} calling: {tool_name} with args: {tool_args}")
                    # Reset timeout while tool is executing
                    start_time = asyncio.get_event_loop().time()
                    continue
                
                elif event_type == "tool_end":
                    # Tool execution completed by SDK - log the result
                    tool_name = event.tool.name if hasattr(event, 'tool') else 'unknown'
                    tool_output = event.output if hasattr(event, 'output') else 'no output'
                    print(f"[TOOL] {tool_name} result: {str(tool_output)[:200]}")
                    # Reset timeout after tool completes
                    start_time = asyncio.get_event_loop().time()
                    continue
                
                elif event_type == "handoff":
                    # Track agent handoff - restart session to enable unique voice per agent
                    from_agent = event.from_agent.name
                    to_agent_obj = event.to_agent
                    to_agent = to_agent_obj.name
                    active_agent = to_agent
                    
                    # Extract the most recent user request from the event history if available
                    # This ensures we capture what triggered the handoff
                    if hasattr(event, 'history') and event.history:
                        for item in reversed(event.history):
                            if hasattr(item, 'role') and item.role == 'user':
                                if hasattr(item, 'content'):
                                    contents = item.content if isinstance(item.content, list) else [item.content]
                                    for content in contents:
                                        transcript = getattr(content, 'transcript', None) or getattr(content, 'text', None)
                                        if transcript:
                                            last_user_request = transcript
                                            print(f"[CONTEXT] Handoff triggered by user request: '{last_user_request}'")
                                            break
                                    break
                    
                    print(f"[INFO] Handoff: {from_agent} → {to_agent}")
                    
                    # Close current session gracefully
                    print(f"[INFO] Closing session to switch voice for {to_agent}...")
                    try:
                        await session.close()
                        await asyncio.sleep(0.5)
                    except Exception as close_err:
                        print(f"[WARN] Error during session close: {close_err}")
                        await asyncio.sleep(0.5)
                    
                    with _SESSION_LOCK:
                        if session_key in _REALTIME_SESSIONS:
                            del _REALTIME_SESSIONS[session_key]
                    
                    # Get new agent's voice settings from persona
                    new_persona = config.AGENT_PERSONAS.get(to_agent, {})
                    new_voice = new_persona.get("voice", "alloy")
                    new_speed = new_persona.get("speed", 1.0)
                    
                    print(f"[INFO] Starting new session with voice '{new_voice}' for {to_agent}")
                    
                    # Build handoff context to embed in agent instructions
                    handoff_context_instruction = ""
                    if last_user_request:
                        handoff_context_instruction = f"""

                        # ACTIVE HANDOFF CONTEXT
                        You just received a handoff from {from_agent}. The user's most recent request was: "{last_user_request}"
                        DO NOT introduce yourself. Respond DIRECTLY to this request using your tools immediately."""
                    
                    # Clone the agent with handoff context embedded in instructions
                    agent_with_context = to_agent_obj.clone(
                        instructions=(to_agent_obj.instructions or "") + handoff_context_instruction
                    )
                    
                    # Create new runner with the context-aware agent
                    from agents.realtime import RealtimeRunner
                    
                    runner = RealtimeRunner(
                        starting_agent=agent_with_context,
                        config={
                            "model_settings": {
                                "model_name": "gpt-4o-realtime-preview-2024-12-17",
                                "modalities": ["audio"],
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
                    
                    # Start new session - SDK handles context internally
                    session = await runner.run(context={"conversation_id": conversation_id})
                    await session.__aenter__()
                    
                    # Store new session
                    with _SESSION_LOCK:
                        _REALTIME_SESSIONS[session_key] = session
                    
                    print(f"[INFO] New session active for {to_agent}")
                    
                    # Clear response collection for new agent's response
                    response_audio_chunks = []
                    response_text = ""
                    last_seen_history_len = 0
                    
                    # Trigger the agent to generate a response immediately
                    # Send brief silence audio then commit to trigger turn detection
                    print(f"[INFO] Triggering {to_agent} to respond to: '{last_user_request[:50]}...'")
                    try:
                        # Send minimal silence (100ms of silence at 24kHz = 2400 samples)
                        silence_samples = np.zeros(2400, dtype=np.int16)
                        silence_bytes = silence_samples.tobytes()
                        await session.send_audio(silence_bytes, commit=True)
                    except Exception as trigger_err:
                        print(f"[WARN] Could not trigger response: {trigger_err}")
                    
                    # Reset timeout and start NEW event loop for the new session
                    # (The original iterator is bound to the old closed session)
                    start_time = asyncio.get_event_loop().time()
                    print(f"[INFO] Listening for {to_agent}'s response...")
                    
                    async for new_event in session:
                        # Check timeout
                        if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                            print("[WARN] Response timeout for new agent")
                            break
                        
                        new_event_type = new_event.type
                        
                        if new_event_type == "audio":
                            response_audio_chunks.append(new_event.audio.data)
                        
                        elif new_event_type == "agent_start":
                            active_agent = new_event.agent.name
                            print(f"[INFO] Agent started: {active_agent}")
                        
                        elif new_event_type == "tool_start":
                            tool_name = new_event.tool.name if hasattr(new_event, 'tool') else 'unknown'
                            tool_args = new_event.arguments if hasattr(new_event, 'arguments') else '{}'
                            print(f"[TOOL] {active_agent} calling: {tool_name} with args: {tool_args}")
                            start_time = asyncio.get_event_loop().time()
                        
                        elif new_event_type == "tool_end":
                            tool_name = new_event.tool.name if hasattr(new_event, 'tool') else 'unknown'
                            tool_output = new_event.output if hasattr(new_event, 'output') else 'no output'
                            print(f"[TOOL] {tool_name} result: {str(tool_output)[:200]}")
                            start_time = asyncio.get_event_loop().time()
                        
                        elif new_event_type == "history_updated":
                            history = new_event.history
                            new_items = history[last_seen_history_len:]
                            last_seen_history_len = len(history)
                            
                            for item in new_items:
                                if hasattr(item, 'role') and hasattr(item, 'content'):
                                    contents = item.content if isinstance(item.content, list) else [item.content]
                                    for content in contents:
                                        if item.role == 'assistant':
                                            transcript = getattr(content, 'text', None) or getattr(content, 'transcript', None)
                                            if transcript:
                                                response_text = transcript
                                                print(f"[TRANSCRIPT] {active_agent} said: {response_text}")
                        
                        elif new_event_type == "audio_end":
                            if response_text:
                                print(f"[SUMMARY] {active_agent} response: '{response_text}'")
                            else:
                                print(f"[WARN] No transcript captured for {active_agent}'s response")
                            break
                        
                        elif new_event_type == "error":
                            print(f"[ERROR] Realtime API error: {new_event.error if hasattr(new_event, 'error') else 'unknown'}")
                            break
                    
                    # After new agent responds, break out of the original loop
                    # (we've already collected their response)
                    break
                
                elif event_type == "agent_start":
                    # Track which agent is responding
                    active_agent = event.agent.name
                    print(f"[INFO] Agent started: {active_agent}")
                
                elif event_type == "history_updated":
                    # Only process NEW history items to avoid duplicates
                    history = event.history
                    new_items = history[last_seen_history_len:]
                    last_seen_history_len = len(history)
                    
                    for item in new_items:
                        if hasattr(item, 'role') and hasattr(item, 'content'):
                            contents = item.content if isinstance(item.content, list) else [item.content]
                            for content in contents:
                                # User input transcript
                                if item.role == 'user':
                                    transcript = getattr(content, 'transcript', None) or getattr(content, 'text', None)
                                    if transcript:
                                        user_transcript = transcript
                                        last_user_request = transcript  # Save for handoff context
                                        print(f"[TRANSCRIPT] User said: {user_transcript}")
                                # Assistant response transcript
                                elif item.role == 'assistant':
                                    transcript = getattr(content, 'text', None) or getattr(content, 'transcript', None)
                                    if transcript:
                                        response_text = transcript
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
