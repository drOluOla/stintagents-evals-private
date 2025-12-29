"""
Gradio UI components for StintAgents Voice AI
"""
import gradio as gr
import numpy as np
import random
from .utils import process_voice_input

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


def create_gradio_interface(CONVERSATION_SESSIONS, conversation_id):
    """Create Gradio interface with centralized layout - AUDIO ONLY"""
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

                /* Hide the music icon */
                #audio_input label svg {
                    display: none !important;
                }

                /* Hide the close X button */
                #audio_input button[aria-label="Clear"] {
                    display: none !important;
                }

                /* Style the entire label container*/
                #audio_input label {
                  color: #10b981 !important;
                  background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, 
                    rgba(5, 150, 105, 0.1) 100%) !important;
                  padding: 8px 20px !important;  
                  border: 1px solid rgba(16, 185, 129, 0.3) !important;
                  text-align: center !important;
                  min-height: 25px !important;  
                  margin: 0 auto !important;
                  display: flex !important;  
                  align-items: center !important;  
                  justify-content: center !important;  
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
        # Conversation state
        conversation_state = gr.State(value=conversation_id)
        
        # Audio buffer for accumulating chunks
        audio_buffer_state = gr.State(value=[])
        silence_counter_state = gr.State(value=0)
        has_speech_state = gr.State(value=False)


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

            # Start Fresh Button
            with gr.Column(elem_classes="audio-recorder-container"):
                audio_input = gr.Audio(
                    label=" ", # Leave blank
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    elem_id="audio_input",
                )
                clear_session_btn = gr.Button(
                    "Reset Session", 
                    variant="secondary",
                    size="lg"
                )

        def detect_silence_and_process(audio_chunk, audio_buffer, silence_counter, has_speech, conversation_id):
                """Detect silence and process complete audio"""
                agent_names = list(config.AGENT_PERSONAS.keys())
                n_avatars = len(agent_names)
                def avatar_updates():
                    return [gr.update() for _ in range(n_avatars)]

                if audio_chunk is None:
                    return (
                        audio_buffer,
                        silence_counter,
                        has_speech,
                        gr.update(),
                        gr.update(),
                        *avatar_updates()
                    )

                sample_rate, audio_data = audio_chunk

                # Ensure audio_data is float for RMS calculation
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

                # Calculate RMS to detect speech/silence
                rms = np.sqrt(np.mean(audio_data ** 2))

                # Thresholds
                speech_threshold = 0.02  # Noise above this is considered speech
                silence_threshold = 0.005  # Noise below this is silence
                required_silence_chunks = 1  # ~2 seconds of silence (at ~0.5s per chunk)

                # Detect if current chunk has speech
                if rms > speech_threshold:
                    has_speech = True
                    silence_counter = 0
                    audio_buffer.append(audio_chunk)

                    if len(audio_buffer) == 1:  # First speech chunk
                        return (
                            audio_buffer,
                            silence_counter,
                            has_speech,
                            gr.update(),
                            gr.update(),
                            *avatar_updates()
                        )
                elif rms < silence_threshold and has_speech:
                    # We have silence after speech
                    silence_counter += 1
                    audio_buffer.append(audio_chunk)

                    # Show "Processing..." when silence threshold is first reached
                    if silence_counter == required_silence_chunks:
                        return (
                            audio_buffer,
                            silence_counter,
                            has_speech,
                            gr.update(label="Processing..."),
                            gr.update(),
                            *avatar_updates()
                        )

                    # If we've had enough silence, process the audio
                    if silence_counter >= required_silence_chunks and len(audio_buffer) > required_silence_chunks:
                        # Concatenate all buffered audio
                        try:
                            all_audio_data = np.concatenate([chunk[1] for chunk in audio_buffer])
                            complete_audio = (sample_rate, all_audio_data)

                            # Process the audio - returns (audio, agent_name)
                            output_audio, active_agent = process_voice_input(
                                complete_audio,
                                conversation_id,
                                runner=config.Runner,
                                hr_manager_agent=config.hr_manager
                            )

                            # Generate avatars based on active agent (dynamic)
                            avatar_htmls = [create_agent_avatar(agent_name, active_agent == agent_name) for agent_name in agent_names]
                            # Reset states
                            return (
                                [],     # Clear buffer
                                0,      # Reset silence counter
                                False,  # Reset has_speech
                                gr.update(label=" "),
                                output_audio,
                                *avatar_htmls
                            )
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            return (
                                [],
                                0,
                                False,
                                gr.update(),
                                gr.update(),
                                *avatar_updates()
                            )
                else:
                    # Silence but no speech yet, or in between
                    if has_speech:
                        audio_buffer.append(audio_chunk)

                # Continue accumulating
                return (
                    audio_buffer,
                    silence_counter,
                    has_speech,
                    gr.update(),
                    gr.update(),
                    *avatar_updates()
                )

        # Dynamically set outputs for avatars
        avatar_output_components = [avatar_components[name] for name in agent_names]
        audio_input.stream(
            fn=detect_silence_and_process,
            inputs=[audio_input, audio_buffer_state, silence_counter_state, has_speech_state, conversation_state],
            outputs=[
                audio_buffer_state,
                silence_counter_state,
                has_speech_state,
                audio_input,
                audio_output,
                *avatar_output_components
            ]
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
                *avatar_htmls,
                [],
                0,
                False
            )

        clear_session_btn.click(
            fn=clear_onboarding_session,
            inputs=[conversation_state],
            outputs=[
                audio_output,
                *avatar_output_components,
                audio_buffer_state,
                silence_counter_state,
                has_speech_state
            ]
        )

    return iface
