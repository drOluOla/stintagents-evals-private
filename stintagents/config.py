"""
Configuration and constants for StintAgents Voice AI
"""


# Default agent personas
_DEFAULT_AGENT_PERSONAS = {
    "HR Manager": {
        "voice": "alloy",
        "speed": 1.0,
        "emoji": "👔",
        "description": "Onboarding Coordinator",
        "specialty": "Company Culture & Benefits",
        "color": "#3b82f6"
    },
    "AI Colleague": {
        "voice": "nova",
        "speed": 1.0,
        "emoji": "🤖",
        "description": "AI Assistant",
        "specialty": "Daily Operations & Support",
        "color": "#8b5cf6"
    },
    "IT Staff": {
        "voice": "echo",
        "speed": 1.0,
        "emoji": "💻",
        "description": "Technical Support",
        "specialty": "Systems & Access",
        "color": "#10b981"
    },
    "Line Manager": {
        "voice": "fable",
        "speed": 1.0,
        "emoji": "📊",
        "description": "Team Lead",
        "specialty": "Goals & Performance",
        "color": "#f59e0b"
    }
}

# Mutable agent personas (can be overridden)
AGENT_PERSONAS = _DEFAULT_AGENT_PERSONAS.copy()

def set_agent_personas(personas_dict):
    """
    Set custom agent personas at runtime.
    Usage: set_agent_personas({ ... })
    """
    global AGENT_PERSONAS
    AGENT_PERSONAS = personas_dict.copy()

def reset_agent_personas():
    """
    Reset agent personas to default.
    """
    global AGENT_PERSONAS
    AGENT_PERSONAS = _DEFAULT_AGENT_PERSONAS.copy()

# Global session storage (will be initialized in notebook)
CONVERSATION_SESSIONS = {}

# Tool expected responses (for evaluation)
CURRENT_TOOL_EXPECTED = {}

# Speech-to-text configuration
# Options: "local" (faster-whisper) or "api" (OpenAI gpt-4o-mini-transcribe)
STT_MODE = "local"

def set_stt_mode(mode: str):
    """
    Set the speech-to-text mode.
    
    Args:
        mode: "local" for faster-whisper (offline, requires download)
              "api" for OpenAI gpt-4o-mini-transcribe (requires API key)
    """
    global STT_MODE
    if mode not in ("local", "api"):
        raise ValueError(f"Invalid STT mode: {mode}. Use 'local' or 'api'.")
    STT_MODE = mode
    print(f"[config] STT mode set to: {mode}")
