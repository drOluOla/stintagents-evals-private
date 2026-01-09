"""
Gradio UI components for StintAgents Voice AI - Realtime API with FastRTC
"""
import gradio as gr
from fastrtc import WebRTC
from .fastrtc_handler import RealtimeAudioHandler

import stintagents.config as config


def create_agent_avatar(agent_name: str, is_speaking: bool = False) -> str:
    """Generate HTML avatar with visual feedback."""
    personas = config.AGENT_PERSONAS
    agent_cfg = personas.get(agent_name, {})
    border = "box-shadow: 0 0 20px #ff4444; border-color: #ff4444; animation: pulse 1s infinite;" if is_speaking else "box-shadow: 0 0 15px #059669; border-color: #059669;"
    overlay = "#ff444420" if is_speaking else "rgba(255, 255, 255, 0.05)"
    return f"""
    <style>@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}</style>
    <div style="text-align: center; padding: 20px; border-radius: 15px; 
                background: linear-gradient(135deg, {agent_cfg.get('color', '#059669')}20, {agent_cfg.get('color', '#059669')}10);
                border: 3px solid {agent_cfg.get('color', '#059669')}; {border} height: 220px; 
                display: flex; flex-direction: column; justify-content: center; position: relative;">
        <div style="position: absolute; inset: 0; background: {overlay}; z-index: 1;"></div>
        <div style="position: relative; z-index: 2;">
            <div style="font-size: 4em; margin-bottom: 10px;">{agent_cfg.get('emoji', '🤖')}</div>
            <div style="font-weight: bold; font-size: 1.2em; color: #fff;">{agent_name}</div>
            <div style="color: #666; font-size: 0.9em;">{agent_cfg.get('description', '')}</div>
            <div style="color: #888; font-size: 0.8em;">{agent_cfg.get('specialty', '')}</div>
        </div>
    </div>"""


def create_gradio_interface(CONVERSATION_SESSIONS, conversation_id, realtime_agent=None):
    """Create Gradio interface with centralized layout - AUDIO ONLY with RealtimeAgent"""
    with gr.Blocks(title="Simulated Multi-Agent Voice Call") as iface:
        # Add CSS using HTML component
        gr.HTML("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');
                * {
                    font-family: 'Roboto Mono', sans-serif !important;
                }

                .agent-grid-container {
                    max-width: 700px !important;
                    margin: 20px auto !important;
                }

                .audio-recorder-container {
                    max-width: 900px !important;
                    margin: 20px auto !important;
                }

                .audio-recorder-container .gradio-audio {
                    width: 100% !important;
                }

                .center-content {
                    max-width: 900px !important;
                    margin: 0 auto !important;
                }

                #audio_output {
                    display: none !important;
                }

                .main-title h1 {
                    text-align: center !important;
                }

                .divider hr {
                    max-width: 900px !important;
                    margin: 20px auto !important;
                }

                /* Reduce margin for instruction text */
                .center-content > .prose {
                    margin-bottom: 5px !important;
                    margin-top: 15px !important;
                }

                /* Style WebRTC component */
                #audio_input {
                    max-width: 900px !important;
                    margin: 0 auto !important;
                }

                /* Style WebRTC label */
                #audio_input label {
                    color: #10b981 !important;
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%,
                        rgba(5, 150, 105, 0.1) 100%) !important;
                    padding: 8px 20px !important;
                    border: 1px solid rgba(16, 185, 129, 0.3) !important;
                    border-radius: 8px !important;
                    text-align: center !important;
                    min-height: 25px !important;
                    margin: 0 auto !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }

                /* Style WebRTC controls */
                #audio_input button {
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%,
                        rgba(5, 150, 105, 0.2) 100%) !important;
                    border: 1px solid rgba(16, 185, 129, 0.4) !important;
                    color: #10b981 !important;
                }

            </style>
        """)
        gr.Markdown("""
          <div style="
              text-align: center;
              padding: 8px 20px;
              margin: 10px auto;
              max-width: 700px;
              background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
              border-radius: 8px;
              border: 1px solid rgba(16, 185, 129, 0.3);
          ">
              <h1 style="
                  font-size: 1.5em;
                  font-weight: 700;
                  color: #10b981;
                  margin: 0;
              ">
                  StintAgents
              </h1>
          </div>
        """)
        # Conversation state and Realtime agent
        conversation_state = gr.State(value=conversation_id)
        realtime_agent_state = gr.State(value=realtime_agent)


        with gr.Column(elem_classes="center-content"):
            # Agent Avatars Grid (dynamic)
            with gr.Column(elem_classes="agent-grid-container"):
                avatar_components = {}
                agent_names = list(config.AGENT_PERSONAS.keys())
                # Display avatars in rows of 2
                for i in range(0, len(agent_names), 2):
                    with gr.Row(equal_height=True):
                        for j in range(2):
                            if i + j < len(agent_names):
                                agent_name = agent_names[i + j]
                                avatar_components[agent_name] = gr.HTML(
                                    value=create_agent_avatar(agent_name),
                                    label=agent_name
                                )

            # Audio Response (hidden but functional for autoplay)
            audio_output = gr.Audio(
                label="Team Response",
                interactive=False,
                autoplay=True,
                elem_id="audio_output"
            )

            # Real-time audio streaming with FastRTC WebRTC
            with gr.Column(elem_classes="audio-recorder-container"):
                # Create FastRTC handler
                audio_handler = RealtimeAudioHandler(
                    conversation_id=conversation_id,
                    realtime_agent=realtime_agent,
                    CONVERSATION_SESSIONS=CONVERSATION_SESSIONS
                )

                # WebRTC component for low-latency audio streaming
                # FastRTC handles streaming automatically via the handler
                audio_input = WebRTC(
                    label=" ",  # Blank label to match original design
                    mode="send-receive",
                    modality="audio",
                    rtc_configuration=None,  # Use default STUN servers
                    elem_id="audio_input",
                    stream_handler=audio_handler  # Handler processes audio automatically
                )

                clear_session_btn = gr.Button(
                    "Reset Session",
                    variant="secondary",
                    size="lg"
                )

        # Dynamically set outputs for avatars
        avatar_output_components = [avatar_components[name] for name in agent_names]

        # Handle additional outputs from FastRTC (avatar updates)
        audio_input.on_additional_outputs(
            lambda *avatars: avatars,
            outputs=avatar_output_components,
            queue=False,
            show_progress="hidden"
        )

        def clear_onboarding_session(conversation_id):
            """Clear onboarding session"""
            if conversation_id in CONVERSATION_SESSIONS:
                CONVERSATION_SESSIONS[conversation_id].close()
                del CONVERSATION_SESSIONS[conversation_id]
            # Dynamically reset avatars
            avatar_htmls = [create_agent_avatar(agent_name) for agent_name in config.AGENT_PERSONAS.keys()]
            return (
                None,
                *avatar_htmls
            )

        clear_session_btn.click(
            fn=clear_onboarding_session,
            inputs=[conversation_state],
            outputs=[
                audio_output,
                *avatar_output_components
            ]
        )

    return iface
