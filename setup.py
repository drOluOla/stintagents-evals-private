from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stintagents-evals-private",
    version="0.1.0",
    author="Oluseyi",
    description="Multi-agent voice interaction system for employee onboarding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drOluOla/stintagents-evals-private",  
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "openai>=1.59.7",
        "openai-agents[voice]==0.4.2",
        "inspect_ai==0.3.150",
        "gradio==5.49.1",
        "fastrtc[vad]>=0.0.34",  # WebRTC for low-latency audio streaming with VAD
        "numpy>=2.1.3",
        "torch>=2.5.1",
        "scipy>=1.14.1",
        # "faster-whisper==1.2.1",
        "pydub-ng>=0.2.0",
        "soundfile>=0.12.1",
        "websockets>=13.0"  # Required for Realtime API
    ],
)
