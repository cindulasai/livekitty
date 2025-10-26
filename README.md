# LiveKit Voice Assistant

A real-time voice interaction system using LiveKit, Whisper STT, Ollama LLM, and browser-based TTS.

## Features

- 🎤 **Speech-to-Text**: OpenAI Whisper for accurate transcription
- 🤖 **LLM Integration**: Ollama (Llama 3) for intelligent responses
- 🔊 **Text-to-Speech**: Browser Web Speech API
- 🔄 **Real-time Communication**: LiveKit for low-latency audio streaming
- 🛡️ **Feedback Prevention**: 3-second cooldown mechanism prevents audio loops

## Prerequisites

1. **Docker** - For running LiveKit server
2. **Python 3.8+** - For the voice agent and token server
3. **Ollama** - Running locally with Llama 3 model
4. **Modern Browser** - Chrome, Edge, or Firefox with Web Speech API support

## Installation

### 1. Start LiveKit Server

```bash
docker run --rm -p 7880:7880 \
  -e LIVEKIT_KEYS="devkey: secret" \
  livekit/livekit-server \
  --dev --node-ip=127.0.0.1
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env if needed (default values should work for local development)
```

### 4. Install Ollama and Model

```bash
# Download and install Ollama from https://ollama.ai
# Then pull the Llama 3 model
ollama pull llama3:8b
```

## Running the Application

### Terminal 1: Start Token Server

```bash
python token_server.py
```

This will start the token generation server on `http://localhost:5000`

### Terminal 2: Start Voice Agent

```bash
python agent/voice_agent.py
```

This will:
- Load the Whisper model
- Connect to LiveKit room
- Start listening for audio

### Terminal 3: Start Web Server

```bash
cd web
python -m http.server 8002
```

### Access the Web Interface

Open your browser to: `http://localhost:8002`

## Usage

1. **Connect**: Enter your name and click "Connect & Start Voice Chat"
2. **Speak**: The microphone will auto-enable - just start talking
3. **Listen**: The assistant's response will be spoken back to you
4. **Conversation**: All interactions are displayed in the conversation panel

## How It Works

1. **User speaks** → Microphone captures audio
2. **LiveKit streams** → Audio sent to voice agent
3. **Whisper transcribes** → Speech converted to text
4. **Ollama processes** → LLM generates response
5. **Response sent** → Data message to browser
6. **Browser speaks** → Web Speech API plays audio
7. **Cooldown** → 3-second pause prevents feedback loop

## Troubleshooting

### No audio detected
- Check microphone permissions in browser
- Ensure microphone is not muted
- Check browser console for errors

### Empty transcriptions
- Speak louder or closer to microphone
- Check that Whisper model loaded successfully
- Verify audio levels in agent logs

### Feedback loop (repeated responses)
- The 3-second cooldown should prevent this
- Ensure browser unpublishes track during TTS
- Check that only one client is connected

### Ollama not responding
- Verify Ollama is running: `ollama list`
- Check model is loaded: `ollama run llama3:8b`
- Verify OLLAMA_BASE_URL in .env

## Configuration

### Environment Variables (.env)

- `LIVEKIT_URL`: WebSocket URL for LiveKit server (default: ws://localhost:7880)
- `LIVEKIT_API_KEY`: API key for LiveKit (default: devkey)
- `LIVEKIT_API_SECRET`: API secret for LiveKit (default: secret)
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434/v1)
- `OLLAMA_MODEL`: Model to use (default: llama3:8b)
- `WHISPER_MODEL`: Whisper model size (default: base, options: tiny/base/small/medium/large)
- `ROOM_NAME`: LiveKit room name (default: voice-room)

### Adjusting Performance

**Faster transcription (less accurate)**:
```bash
WHISPER_MODEL=tiny
```

**Better transcription (slower)**:
```bash
WHISPER_MODEL=medium
```

**Shorter responses**:
Edit `token_server.py` and `agent/voice_agent.py`, change `max_tokens` to 50-100

## Architecture

```
┌─────────────┐     Audio Stream      ┌──────────────┐
│   Browser   │ ←──────────────────→  │   LiveKit    │
│  (Web UI)   │                        │    Server    │
└─────────────┘                        └──────────────┘
      │                                       ▲
      │ Token Request                         │
      ▼                                       │ Audio
┌─────────────┐                               │
│   Token     │                               │
│   Server    │                        ┌──────────────┐
└─────────────┘                        │    Voice     │
      │                                │    Agent     │
      │ LLM Request                    └──────────────┘
      ▼                                       │
┌─────────────┐                               │
│   Ollama    │ ←─────────────────────────────┘
│   (LLM)     │       Response
└─────────────┘
```

## License

MIT

## Credits

- LiveKit: https://livekit.io
- OpenAI Whisper: https://github.com/openai/whisper
- Ollama: https://ollama.ai
