"""
FastRTC StreamHandler for StintAgents Realtime API Integration
"""
import numpy as np
from fastrtc import StreamHandler
import stintagents.config as config
from typing import Optional


class RealtimeAudioHandler(StreamHandler):
    """
    FastRTC StreamHandler that processes audio through OpenAI Realtime API.
    Handles audio streaming, agent responses, and multi-agent handoffs.
    """

    def __init__(self, conversation_id: str, realtime_agent, CONVERSATION_SESSIONS):
        """
        Initialize the handler with conversation context.

        Args:
            conversation_id: Unique identifier for this conversation
            realtime_agent: RealtimeAgent instance from agents.realtime
            CONVERSATION_SESSIONS: Global session storage dictionary
        """
        self.conversation_id = conversation_id
        self.realtime_agent = realtime_agent
        self.CONVERSATION_SESSIONS = CONVERSATION_SESSIONS
        self.active_agent_name = realtime_agent.name if realtime_agent else None

        # Import the streaming utility
        from .utils import stream_audio_chunk_realtime
        self.stream_audio_chunk = stream_audio_chunk_realtime

    def copy(self):
        """Create a copy of this handler for a new connection."""
        return RealtimeAudioHandler(
            self.conversation_id,
            self.realtime_agent,
            self.CONVERSATION_SESSIONS
        )

    def receive(self, frame: tuple[int, np.ndarray]) -> Optional[tuple[int, np.ndarray]]:
        """
        Process incoming audio frame and return response if available.

        Args:
            frame: Tuple of (sample_rate, audio_data)

        Returns:
            Tuple of (sample_rate, audio_response) if response available, else None
        """
        if frame is None:
            return None

        # Stream the audio chunk to Realtime API
        output_audio, active_agent = self.stream_audio_chunk(
            frame,
            self.conversation_id,
            realtime_agent=self.realtime_agent
        )

        # Update active agent if changed
        if active_agent:
            self.active_agent_name = active_agent

        # Return audio response if available
        if output_audio is not None:
            return output_audio

        return None

    def get_active_agent(self) -> Optional[str]:
        """Get the name of the currently active agent."""
        return self.active_agent_name
