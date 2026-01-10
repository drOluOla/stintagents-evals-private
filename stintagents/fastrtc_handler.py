"""
FastRTC Handler for StintAgents Realtime API Integration
Uses ReplyOnPause for automatic voice activity detection and turn-taking
"""
import numpy as np
from fastrtc import AdditionalOutputs
import stintagents.config as config
from typing import Tuple, Optional


def create_realtime_handler(conversation_id: str, realtime_agent, CONVERSATION_SESSIONS):
    """
    Create a handler function for FastRTC ReplyOnPause.

    This generator function processes audio through OpenAI Realtime API
    and yields responses with avatar updates.

    Args:
        conversation_id: Unique identifier for this conversation
        realtime_agent: RealtimeAgent instance from agents.realtime
        CONVERSATION_SESSIONS: Global session storage dictionary

    Returns:
        Generator function that processes audio and yields responses
    """
    # Import the streaming utility
    from .utils import process_voice_input_realtime
    from .ui import create_agent_avatar

    def handler(audio_data: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
        """
        Process audio input and return AI response.

        This function is called by ReplyOnPause after detecting user finished speaking.

        Args:
            audio_data: Tuple of (sample_rate, audio_array) from user's microphone

        Yields:
            Tuple of (sample_rate, response_audio) with AdditionalOutputs for avatars
        """
        if audio_data is None:
            return

        # Process through Realtime API (waits for complete response)
        output_audio, active_agent = process_voice_input_realtime(
            audio_data,
            conversation_id=conversation_id,
            realtime_agent=realtime_agent
        )

        # If we got a response, yield it with avatar updates
        if output_audio is not None and active_agent is not None:
            # Create avatar updates showing which agent is speaking
            agent_names = list(config.AGENT_PERSONAS.keys())
            avatar_htmls = [
                create_agent_avatar(agent_name, active_agent == agent_name)
                for agent_name in agent_names
            ]

            # Yield additional outputs for avatar highlighting
            yield AdditionalOutputs(*avatar_htmls)

            # Return the audio response
            yield output_audio

    return handler
