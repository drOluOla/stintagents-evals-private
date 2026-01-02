# Quick Start Guide - Realtime API

## Installation

```bash
pip install git+https://github.com/drOluOla/stintagents-evals-private.git
pip install websockets
```

## Basic Setup

```python
import os
from stintagents import create_realtime_session, create_gradio_interface
from agents import SQLiteSession, function_tool

# Set API Key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Define a simple tool
@function_tool
def greet_user(name: str) -> str:
    """Greet the user by name."""
    return f"Hello {name}! Welcome to StintAgents."

# Initialize session
conversation_id = "my_session"
CONVERSATION_SESSIONS = {}
CONVERSATION_SESSIONS[conversation_id] = SQLiteSession(conversation_id)

# Create Realtime session
realtime_session = create_realtime_session(
    conversation_id=conversation_id,
    agent_name="Assistant",
    instructions="You are a helpful assistant. Greet users warmly.",
    tools=[greet_user],
    tool_handlers={
        'greet_user': lambda name: greet_user.func(name)
    }
)

# Launch UI
iface = create_gradio_interface(CONVERSATION_SESSIONS, conversation_id, realtime_session)
iface.launch(share=True)
```

## Key Differences from Old Version

### Old Way (STT → LLM → TTS)
```python
from stintagents import create_gradio_interface

# Used Runner and Agents SDK
config.Runner = Runner
config.hr_manager = hr_manager_agent

iface = create_gradio_interface(CONVERSATION_SESSIONS, conversation_id)
```

### New Way (Realtime API)
```python
from stintagents import create_realtime_session, create_gradio_interface

# Create Realtime session with tools
realtime_session = create_realtime_session(
    conversation_id=conversation_id,
    agent_name="HR Manager",
    instructions="Your instructions here...",
    tools=[tool1, tool2],
    tool_handlers={
        'tool1': lambda arg: tool1.func(arg),
        'tool2': lambda: tool2.func()
    }
)

iface = create_gradio_interface(CONVERSATION_SESSIONS, conversation_id, realtime_session)
```

## Tool Registration Pattern

```python
# Define tool
@function_tool
def get_info(topic: str) -> str:
    """Get information about a topic."""
    return f"Here's info about {topic}"

# Register with Realtime session
tools = [get_info]
tool_handlers = {
    'get_info': lambda topic: get_info.func(topic)
}

realtime_session = create_realtime_session(
    conversation_id="session_id",
    tools=tools,
    tool_handlers=tool_handlers
)
```

## Multiple Tools Example

```python
@function_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"

@function_tool
def get_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M")

# Register all tools
tools = [get_weather, get_time]
tool_handlers = {
    'get_weather': lambda city: get_weather.func(city),
    'get_time': lambda: get_time.func()
}

realtime_session = create_realtime_session(
    conversation_id="multi_tool_session",
    agent_name="Weather Bot",
    instructions="Help users with weather and time info.",
    tools=tools,
    tool_handlers=tool_handlers
)
```

## Custom Voice Personas

```python
from stintagents.config import set_agent_personas

# Define custom personas
set_agent_personas({
    "Weather Bot": {
        "voice": "shimmer",
        "speed": 1.1,
        "description": "Energetic & Informative",
        "emoji": "🌤️",
        "color": "#FFD700"
    }
})

# Use in Realtime session
realtime_session = create_realtime_session(
    conversation_id="weather_session",
    agent_name="Weather Bot"  # Will use shimmer voice
)
```

## Troubleshooting

### Error: WebSocket connection failed
**Solution**: Check your API key and ensure it has Realtime API access
```python
import os
print(os.environ.get("OPENAI_API_KEY")[:10] + "...")  # Verify key is set
```

### Error: No audio output
**Solution**: Ensure your browser allows microphone access and audio autoplay

### Error: Tool not being called
**Solution**: Check that tool handler name matches function name exactly
```python
# Correct
tool_handlers = {
    'get_weather': lambda city: get_weather.func(city)  # Name matches function
}

# Incorrect
tool_handlers = {
    'weather': lambda city: get_weather.func(city)  # Name mismatch!
}
```

## Performance Tips

1. **Use PCM16 @ 24kHz**: Input audio is automatically converted, but providing the right format reduces processing
2. **Keep instructions concise**: Shorter system prompts improve response time
3. **Batch tool registrations**: Register all tools at session creation
4. **Reuse sessions**: Don't create new sessions for each interaction

## Next Steps

- See [REALTIME_API_MIGRATION.md](REALTIME_API_MIGRATION.md) for detailed migration guide
- Check [StintAgents_Evals_Safety_Private.ipynb](StintAgents_Evals_Safety_Private.ipynb) for complete example
- Review [utils.py](stintagents/utils.py) for API reference

## Support

For issues or questions, please open an issue on GitHub.

---
**Version**: 2.0.0  
**Last Updated**: January 2, 2026
