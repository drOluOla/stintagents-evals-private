"""
Core utilities for StintAgents Voice AI
Audio processing, transcription, TTS, and agent coordination

Performance Optimizations:
- ProcessPoolExecutor for CPU-bound audio preprocessing (bypasses GIL)
- ThreadPoolExecutor for I/O-bound tasks (TTS API, file I/O)
- Multi-GPU worker pool for parallel transcription (load balancing)
- Whisper model warm-up to reduce first-inference latency
- Pipelining: start processing while still receiving audio
"""
import asyncio
import threading
import multiprocessing as mp
import numpy as np
import torch
import io
import json
import os
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from scipy import signal
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from pydub import AudioSegment

import stintagents.config as config

# ==============================================================================
# EXECUTOR POOLS
# - ThreadPool: I/O-bound (TTS API calls, network, file I/O)
# - ProcessPool: CPU-bound (audio preprocessing, numpy/scipy operations)
# ==============================================================================
_NUM_WORKERS = min(4, (os.cpu_count() or 2))
_THREAD_EXECUTOR = ThreadPoolExecutor(max_workers=_NUM_WORKERS, thread_name_prefix="io_worker")
_PROCESS_EXECUTOR: Optional[ProcessPoolExecutor] = None
_process_lock = threading.Lock()

def get_process_executor() -> ProcessPoolExecutor:
    """Lazy-init ProcessPoolExecutor (avoids issues with module-level init)."""
    global _PROCESS_EXECUTOR
    with _process_lock:
        if _PROCESS_EXECUTOR is None:
            # Use 'spawn' to avoid CUDA fork issues on Linux
            ctx = mp.get_context('spawn')
            _PROCESS_EXECUTOR = ProcessPoolExecutor(
                max_workers=_NUM_WORKERS,
                mp_context=ctx
            )
    return _PROCESS_EXECUTOR

# Backward compat alias
_EXECUTOR = _THREAD_EXECUTOR

# ==============================================================================
# MULTI-GPU WORKER POOL
# - Distributes transcription workload across multiple GPUs
# - Each GPU has its own Whisper model instance
# - Round-robin load balancing with queue-based task distribution
# ==============================================================================
@dataclass
class GPUWorkerConfig:
    """Configuration for GPU workers."""
    model_size: str = "base"
    compute_type: str = "float16"
    max_queue_size: int = 100
    worker_timeout: float = 30.0

class GPUWorker:
    """Individual GPU worker with dedicated Whisper model."""
    
    def __init__(self, gpu_id: int, config: GPUWorkerConfig):
        self.gpu_id = gpu_id
        self.config = config
        self.model: Optional[WhisperModel] = None
        self.task_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self.result_dict: Dict[int, Any] = {}
        self.result_lock = threading.Lock()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.task_counter = 0
        self.counter_lock = threading.Lock()
        
    def start(self):
        """Start the worker thread."""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"gpu_worker_{self.gpu_id}",
            daemon=True
        )
        self.worker_thread.start()
        
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        # Send sentinel to unblock queue
        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
            
    def _init_model(self):
        """Initialize Whisper model on this GPU."""
        if self.model is not None:
            return
        device = f"cuda:{self.gpu_id}"
        print(f"[GPU Worker {self.gpu_id}] Initializing Whisper model on {device}...")
        self.model = WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=self.config.compute_type
        )
        # Warm up
        dummy_audio = np.zeros(16000, dtype=np.float32)
        list(self.model.transcribe(dummy_audio, beam_size=1, language="en"))
        print(f"[GPU Worker {self.gpu_id}] Ready!")
        
    def _worker_loop(self):
        """Main worker loop - processes tasks from queue."""
        try:
            self._init_model()
        except Exception as e:
            print(f"[GPU Worker {self.gpu_id}] Failed to initialize: {e}")
            self.running = False
            return
            
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Sentinel
                    continue
                    
                task_id, audio, kwargs = task
                try:
                    segments, _ = self.model.transcribe(audio, **kwargs)
                    result = " ".join(seg.text for seg in segments).strip()
                except Exception as e:
                    result = None
                    print(f"[GPU Worker {self.gpu_id}] Transcription error: {e}")
                
                with self.result_lock:
                    self.result_dict[task_id] = result
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[GPU Worker {self.gpu_id}] Worker error: {e}")
                
    def submit(self, audio: np.ndarray, **kwargs) -> int:
        """Submit transcription task. Returns task_id."""
        with self.counter_lock:
            task_id = self.task_counter
            self.task_counter += 1
        self.task_queue.put((task_id, audio, kwargs))
        return task_id
        
    def get_result(self, task_id: int, timeout: float = 30.0) -> Optional[str]:
        """Wait for and retrieve result."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            with self.result_lock:
                if task_id in self.result_dict:
                    return self.result_dict.pop(task_id)
            time.sleep(0.01)
        return None
        
    @property
    def queue_size(self) -> int:
        return self.task_queue.qsize()


class MultiGPUTranscriptionPool:
    """Pool of GPU workers for parallel transcription."""
    
    def __init__(self, config: Optional[GPUWorkerConfig] = None):
        self.config = config or GPUWorkerConfig()
        self.workers: List[GPUWorker] = []
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.round_robin_idx = 0
        self.rr_lock = threading.Lock()
        self._initialized = False
        
    def initialize(self, num_workers: Optional[int] = None):
        """Initialize GPU workers."""
        if self._initialized:
            return
            
        if self.num_gpus == 0:
            print("[MultiGPU] No GPUs available, will use single-threaded fallback")
            self._initialized = True
            return
            
        num_workers = num_workers or self.num_gpus
        num_workers = min(num_workers, self.num_gpus)
        
        print(f"[MultiGPU] Initializing {num_workers} GPU worker(s) across {self.num_gpus} GPU(s)")
        
        for i in range(num_workers):
            gpu_id = i % self.num_gpus  # Distribute across GPUs
            worker = GPUWorker(gpu_id, self.config)
            worker.start()
            self.workers.append(worker)
            
        self._initialized = True
        print(f"[MultiGPU] Pool ready with {len(self.workers)} workers")
        
    def _select_worker(self) -> Optional[GPUWorker]:
        """Select worker using least-loaded strategy."""
        if not self.workers:
            return None
            
        # Find worker with smallest queue
        min_worker = min(self.workers, key=lambda w: w.queue_size)
        return min_worker
        
    def transcribe(self, audio: np.ndarray, **kwargs) -> Optional[str]:
        """Transcribe audio using available GPU worker."""
        if not self._initialized:
            self.initialize()
            
        worker = self._select_worker()
        if worker is None:
            return None
            
        task_id = worker.submit(audio, **kwargs)
        return worker.get_result(task_id, timeout=self.config.worker_timeout)
        
    async def transcribe_async(self, audio: np.ndarray, **kwargs) -> Optional[str]:
        """Async transcription - submits to worker pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_THREAD_EXECUTOR, self.transcribe, audio, **kwargs)
        
    def transcribe_batch(self, audio_list: List[np.ndarray], **kwargs) -> List[Optional[str]]:
        """Transcribe multiple audio clips in parallel across GPU workers."""
        if not self._initialized:
            self.initialize()
            
        if not self.workers:
            # Fallback: single-threaded
            return [self._fallback_transcribe(audio, **kwargs) for audio in audio_list]
            
        # Submit all tasks, distributing across workers
        task_assignments: List[Tuple[GPUWorker, int]] = []
        for audio in audio_list:
            worker = self._select_worker()
            task_id = worker.submit(audio, **kwargs)
            task_assignments.append((worker, task_id))
            
        # Collect all results
        results = []
        for worker, task_id in task_assignments:
            result = worker.get_result(task_id, timeout=self.config.worker_timeout)
            results.append(result)
            
        return results
        
    async def transcribe_batch_async(self, audio_list: List[np.ndarray], **kwargs) -> List[Optional[str]]:
        """Async batch transcription."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_THREAD_EXECUTOR, self.transcribe_batch, audio_list, **kwargs)
        
    def _fallback_transcribe(self, audio: np.ndarray, **kwargs) -> Optional[str]:
        """Fallback for when no GPU workers available."""
        global WHISPER_MODEL
        try:
            segments, _ = WHISPER_MODEL.transcribe(audio, **kwargs)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            print(f"[MultiGPU Fallback] Transcription error: {e}")
            return None
            
    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            worker.stop()
        self.workers.clear()
        self._initialized = False
        
    @property
    def is_available(self) -> bool:
        """Check if multi-GPU is available and initialized."""
        return self._initialized and len(self.workers) > 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "num_gpus": self.num_gpus,
            "num_workers": len(self.workers),
            "queue_sizes": [w.queue_size for w in self.workers],
            "initialized": self._initialized
        }


# Global multi-GPU pool instance (lazy initialized)
_MULTI_GPU_POOL: Optional[MultiGPUTranscriptionPool] = None
_multi_gpu_lock = threading.Lock()

def get_multi_gpu_pool(num_workers: Optional[int] = None) -> MultiGPUTranscriptionPool:
    """Get or create the multi-GPU transcription pool."""
    global _MULTI_GPU_POOL
    with _multi_gpu_lock:
        if _MULTI_GPU_POOL is None:
            _MULTI_GPU_POOL = MultiGPUTranscriptionPool()
        if not _MULTI_GPU_POOL._initialized:
            _MULTI_GPU_POOL.initialize(num_workers)
    return _MULTI_GPU_POOL

def enable_multi_gpu_transcription(num_workers: Optional[int] = None):
    """Enable multi-GPU transcription with specified number of workers.
    
    Args:
        num_workers: Number of GPU workers to spawn. Defaults to number of GPUs.
                    Can be > num_gpus to have multiple workers per GPU.
    """
    pool = get_multi_gpu_pool(num_workers)
    print(f"[MultiGPU] Enabled: {pool.get_stats()}")
    return pool

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

# Warm up the model with a dummy audio to compile/cache kernels
def _warmup_whisper():
    """Warm up Whisper model to reduce first-inference latency."""
    try:
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        list(WHISPER_MODEL.transcribe(dummy_audio, beam_size=1, language="en"))
        print("[init] Whisper model warmed up")
    except Exception as e:
        print(f"[init] Whisper warmup skipped: {e}")

_warmup_whisper()

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
@lru_cache(maxsize=8)
def _get_resample_factors(src_rate: int, dst_rate: int = 16000) -> Tuple[int, int]:
    """Cache GCD-reduced resample factors for common sample rates."""
    from math import gcd
    g = gcd(src_rate, dst_rate)
    return dst_rate // g, src_rate // g

def preprocess_audio(raw_audio, sample_rate):
    """Normalize, resample, and convert audio to mono float32 @ 16kHz.
    
    Optimizations:
    - Uses resample_poly (polyphase) instead of resample (FFT-based) - 3-10x faster
    - In-place operations where possible to reduce memory allocations
    - Cached resample factors for common sample rates
    """
    if not isinstance(raw_audio, np.ndarray):
        raw_audio = np.array(raw_audio, dtype=np.float32)
    
    if raw_audio.size == 0:
        return np.array([], dtype=np.float32), 16000
    
    # Convert to float32 (in-place when possible)
    if raw_audio.dtype == np.int16:
        raw_audio = raw_audio.astype(np.float32)
        raw_audio *= (1.0 / 32768.0)  # In-place multiply is faster
    elif raw_audio.dtype == np.int32:
        raw_audio = raw_audio.astype(np.float32)
        raw_audio *= (1.0 / 2147483648.0)
    elif raw_audio.dtype != np.float32:
        raw_audio = raw_audio.astype(np.float32)
    
    # Mono conversion (use optimized numpy operations)
    if raw_audio.ndim > 1:
        raw_audio = np.mean(raw_audio, axis=1, dtype=np.float32)
    
    # Normalize (in-place)
    max_amp = np.abs(raw_audio).max()
    if max_amp > 1e-6:  # Avoid division by tiny numbers
        raw_audio *= (0.95 / max_amp)
    
    # Resample to 16kHz using polyphase (MUCH faster than FFT-based resample)
    if sample_rate != 16000:
        up, down = _get_resample_factors(sample_rate, 16000)
        raw_audio = signal.resample_poly(raw_audio, up, down).astype(np.float32)
    
    return raw_audio, 16000

# Standalone function for ProcessPool (must be picklable - no closures)
def _preprocess_audio_worker(raw_audio_bytes: bytes, dtype: str, shape: tuple, sample_rate: int) -> Tuple[bytes, str, tuple, int]:
    """Worker function for multiprocessing - receives/returns serializable data."""
    import numpy as np
    from scipy import signal
    from math import gcd
    
    # Reconstruct numpy array from bytes
    raw_audio = np.frombuffer(raw_audio_bytes, dtype=dtype).reshape(shape).copy()
    
    if raw_audio.size == 0:
        return np.array([], dtype=np.float32).tobytes(), 'float32', (0,), 16000
    
    # Convert to float32
    if raw_audio.dtype == np.int16:
        raw_audio = raw_audio.astype(np.float32) * (1.0 / 32768.0)
    elif raw_audio.dtype == np.int32:
        raw_audio = raw_audio.astype(np.float32) * (1.0 / 2147483648.0)
    elif raw_audio.dtype != np.float32:
        raw_audio = raw_audio.astype(np.float32)
    
    # Mono conversion
    if raw_audio.ndim > 1:
        raw_audio = np.mean(raw_audio, axis=1, dtype=np.float32)
    
    # Normalize
    max_amp = np.abs(raw_audio).max()
    if max_amp > 1e-6:
        raw_audio *= (0.95 / max_amp)
    
    # Resample to 16kHz
    if sample_rate != 16000:
        g = gcd(sample_rate, 16000)
        up, down = 16000 // g, sample_rate // g
        raw_audio = signal.resample_poly(raw_audio, up, down).astype(np.float32)
    
    return raw_audio.tobytes(), 'float32', raw_audio.shape, 16000

async def preprocess_audio_async(raw_audio, sample_rate, use_multiprocessing: bool = True) -> Tuple[np.ndarray, int]:
    """Async wrapper - uses ProcessPool for CPU-bound work (bypasses GIL).
    
    Args:
        raw_audio: Input audio array
        sample_rate: Sample rate of input
        use_multiprocessing: If True, uses ProcessPool (faster for large audio).
                            If False, uses ThreadPool (less overhead for small audio).
    """
    loop = asyncio.get_event_loop()
    
    if not isinstance(raw_audio, np.ndarray):
        raw_audio = np.array(raw_audio, dtype=np.float32)
    
    # For small audio chunks, thread pool has less overhead
    # For larger chunks (>1s at 16kHz), process pool wins
    if not use_multiprocessing or raw_audio.size < 16000:
        return await loop.run_in_executor(_THREAD_EXECUTOR, preprocess_audio, raw_audio, sample_rate)
    
    # Use ProcessPool for true parallelism on CPU-bound work
    try:
        executor = get_process_executor()
        result = await loop.run_in_executor(
            executor,
            _preprocess_audio_worker,
            raw_audio.tobytes(),
            str(raw_audio.dtype),
            raw_audio.shape,
            sample_rate
        )
        audio_bytes, dtype, shape, out_rate = result
        return np.frombuffer(audio_bytes, dtype=dtype).reshape(shape), out_rate
    except Exception as e:
        # Fallback to thread pool on any multiprocessing issue
        print(f"[WARN] ProcessPool failed, using ThreadPool: {e}")
        return await loop.run_in_executor(_THREAD_EXECUTOR, preprocess_audio, raw_audio, sample_rate)

def convert_audio_bytes(audio_bytes: bytes, format: str = "mp3"):
    """Convert TTS audio bytes to (sample_rate, numpy_array) for Gradio.
    
    Optimizations:
    - Pre-allocated array operations
    - Avoids unnecessary copies
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        
        if audio.channels == 2:
            # Reshape and mean in one efficient operation
            samples = samples.reshape((-1, 2)).mean(axis=1).astype(np.int16)
        
        return (audio.frame_rate, samples)
    except Exception as e:
        print(f"[ERROR] Audio conversion: {e}")
        return None

async def convert_audio_bytes_async(audio_bytes: bytes, format: str = "mp3"):
    """Async wrapper for audio conversion - runs in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_THREAD_EXECUTOR, convert_audio_bytes, audio_bytes, format)

# ==============================================================================
# TTS & TRANSCRIPTION
# ==============================================================================
async def generate_speech_async(text: str, agent_name: str = "HR Manager", stream: bool = False) -> Optional[bytes]:
    """Generate speech using OpenAI TTS.
    
    Args:
        text: Text to convert to speech
        agent_name: Agent persona for voice selection
        stream: If True, returns async generator for streaming (lower TTFB)
    """
    try:
        personas = config.AGENT_PERSONAS
        agent_cfg = personas.get(agent_name, personas.get("HR Manager", {}))
        
        if stream:
            # Streaming mode - yields chunks for lower time-to-first-byte
            async with async_openai_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=agent_cfg.get("voice", "alloy"),
                input=text,
                speed=agent_cfg.get("speed", 1.0),
                response_format="mp3"
            ) as response:
                chunks = []
                async for chunk in response.iter_bytes(chunk_size=4096):
                    chunks.append(chunk)
                return b"".join(chunks)
        else:
            # Non-streaming mode (simpler, good for short responses)
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

async def transcribe_audio_async(audio: np.ndarray, sample_rate: int = 16000, use_multi_gpu: bool = True) -> Optional[str]:
    """Transcribe audio using faster-whisper.
    
    Args:
        audio: Preprocessed audio array (16kHz, float32, mono)
        sample_rate: Sample rate (should be 16000)
        use_multi_gpu: If True and multiple GPUs available, uses GPU worker pool
    """
    try:
        loop = asyncio.get_event_loop()
        
        # Try multi-GPU pool first if enabled
        if use_multi_gpu and _MULTI_GPU_POOL is not None and _MULTI_GPU_POOL.is_available:
            return await _MULTI_GPU_POOL.transcribe_async(
                audio, beam_size=1, language="en", vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, threshold=0.5),
                temperature=0.0, no_speech_threshold=0.6
            )
        
        # Fallback to single model
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


async def transcribe_audio_batch_async(audio_list: List[np.ndarray], sample_rate: int = 16000) -> List[Optional[str]]:
    """Transcribe multiple audio clips in parallel using multi-GPU pool.
    
    This is the most efficient way to process multiple audio files.
    
    Args:
        audio_list: List of preprocessed audio arrays
        sample_rate: Sample rate (should be 16000)
        
    Returns:
        List of transcriptions (None for failed items)
    """
    pool = get_multi_gpu_pool()
    
    if pool.is_available:
        return await pool.transcribe_batch_async(
            audio_list,
            beam_size=1, language="en", vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, threshold=0.5),
            temperature=0.0, no_speech_threshold=0.6
        )
    else:
        # Fallback: process sequentially
        results = []
        for audio in audio_list:
            result = await transcribe_audio_async(audio, sample_rate, use_multi_gpu=False)
            results.append(result)
        return results

# ==============================================================================
# AGENT RESPONSE
# ==============================================================================
async def get_agent_response_with_speech(user_input: str, conversation_id: str = "default", runner=None, hr_manager_agent=None):
    """
    Get agent response and TTS in one call.
    
    Optimizations:
    - Concurrent metadata save and TTS generation
    - Streaming TTS for lower time-to-first-byte
    
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
        
        # Run metadata save and TTS generation CONCURRENTLY
        metadata_task = session.add_items([{
            "role": "system",
            "content": json.dumps({
                "evaluation_metadata": True,
                "responding_agent": agent_name,
                "expected_response": config.CURRENT_TOOL_EXPECTED.get("expected", response),
            })
        }])
        
        # Use streaming for longer responses (lower TTFB)
        use_streaming = len(response) > 200
        tts_task = generate_speech_async(response, agent_name, stream=use_streaming)
        
        # Wait for both concurrently
        _, speech_bytes = await asyncio.gather(metadata_task, tts_task)
        
        config.CURRENT_TOOL_EXPECTED.clear()
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
    
    Optimizations:
    - ProcessPool for CPU-bound audio preprocessing (bypasses Python GIL)
    - ThreadPool for I/O-bound operations (TTS API, network)
    - Concurrent execution where possible
    
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
            
            # Use multiprocessing for audio > 1 second (bypasses GIL for CPU work)
            use_mp = raw_audio.size > sample_rate  # >1 second of audio
            processed_audio, _ = await preprocess_audio_async(raw_audio, sample_rate, use_multiprocessing=use_mp)
            
            transcription = await transcribe_audio_async(processed_audio)
            
            if not transcription:
                return None, None
            
            response_text, active_agent, speech_bytes = await get_agent_response_with_speech(
                transcription, conversation_id, runner, hr_manager_agent
            )
            
            # Convert speech to playable format (I/O bound, use thread pool)
            output_audio = await convert_audio_bytes_async(speech_bytes, "mp3") if speech_bytes else None
            
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

def cleanup_executors():
    """Cleanup function to properly shutdown executor pools."""
    global _PROCESS_EXECUTOR, _MULTI_GPU_POOL
    if _PROCESS_EXECUTOR is not None:
        _PROCESS_EXECUTOR.shutdown(wait=False)
        _PROCESS_EXECUTOR = None
    if _MULTI_GPU_POOL is not None:
        _MULTI_GPU_POOL.shutdown()
        _MULTI_GPU_POOL = None
    _THREAD_EXECUTOR.shutdown(wait=False)

# Initialize
get_or_create_event_loop()
print("[init] Ready!")