"""
FastRTC StreamHandler for StintAgents Realtime API Integration
"""
import numpy as np
from queue import Queue, Empty
from fastrtc import StreamHandler, AdditionalOutputs
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
        super().__init__()
        self.conversation_id = conversation_id
        self.realtime_agent = realtime_agent
        self.CONVERSATION_SESSIONS = CONVERSATION_SESSIONS
        self.active_agent_name = realtime_agent.name if realtime_agent else None

        # Queue to store responses for emission
        self.response_queue = Queue()

        # Import the streaming utility
        from .utils import stream_audio_chunk_realtime
        self.stream_audio_chunk = stream_audio_chunk_realtime

    def copy(self) -> "RealtimeAudioHandler":
        """Create a copy of this handler for a new connection."""
        return RealtimeAudioHandler(
            self.conversation_id,
            self.realtime_agent,
            self.CONVERSATION_SESSIONS
        )

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """
        Process incoming audio frame from the client.

        This method receives audio from the user's microphone, processes it through
        the Realtime API, and queues any responses for emission.

        Args:
            frame: Tuple of (sample_rate, audio_data)
        """
        if frame is None:
            return

        # Stream the audio chunk to Realtime API
        output_audio, active_agent = self.stream_audio_chunk(
            frame,
            self.conversation_id,
            realtime_agent=self.realtime_agent
        )

        # Update active agent if changed
        if active_agent:
            self.active_agent_name = active_agent

        # Queue response with agent information if available
        if output_audio is not None:
            self.response_queue.put((output_audio, active_agent))

    def emit(self) -> Optional[tuple[int, np.ndarray]]:
        """
        Emit audio responses back to the client.

        Also yields AdditionalOutputs with active agent information for UI updates.

        Returns:
            Tuple of (sample_rate, audio_response) if available, else None
        """
        try:
            # Non-blocking get - return immediately if no response available
            output_audio, active_agent = self.response_queue.get_nowait()

            # Yield additional outputs for avatar updates
            if active_agent:
                # Import here to avoid circular dependency
                from .ui import create_agent_avatar

                agent_names = list(config.AGENT_PERSONAS.keys())
                avatar_htmls = [
                    create_agent_avatar(agent_name, active_agent == agent_name)
                    for agent_name in agent_names
                ]

                # Return audio and yield additional outputs for avatars
                yield AdditionalOutputs(*avatar_htmls)

            return output_audio

        except Empty:
            return None

    def start_up(self) -> None:
        """Called when the stream starts."""
        pass

    def shutdown(self) -> None:
        """Called when the stream ends."""
        pass

    def get_active_agent(self) -> Optional[str]:
        """Get the name of the currently active agent."""
        return self.active_agent_name
