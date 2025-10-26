"""
LiveKit Voice Agent V2 - Robust Implementation
Handles STT (Whisper), LLM (Ollama), TTS (pyttsx3) with proper audio processing
"""

import asyncio
import logging
import os
import numpy as np
import threading
import queue
from collections import deque
import json

from dotenv import load_dotenv
from livekit import rtc, api
import whisper
import requests
import pyttsx3

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAgent:
    """Voice agent with STT, LLM, and TTS capabilities"""
    
    def __init__(self):
        # Initialize Whisper
        whisper_model_name = os.getenv("WHISPER_MODEL", "base")
        logger.info(f"🔄 Loading Whisper model '{whisper_model_name}'...")
        self.whisper_model = whisper.load_model(whisper_model_name)
        logger.info(f"✅ Whisper model loaded")
        
        # Ollama configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3:8b")
        
        # Initialize TTS in separate thread (not used, but kept for compatibility)
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        logger.info(f"✅ TTS engine initialized")
        
        # Audio processing settings
        self.sample_rate = 48000  # LiveKit default
        self.channels = 1
        
    def _tts_worker(self):
        """TTS worker thread"""
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 1.0)
        
        while True:
            try:
                text = self.tts_queue.get()
                if text is None:
                    break
                logger.info(f"🔊 Speaking: {text}")
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"❌ TTS error: {e}")
    
    def speak(self, text: str):
        """Queue text for TTS"""
        self.tts_queue.put(text)
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Calculate duration
            duration = len(audio_data) / self.sample_rate
            logger.info(f"🎤 Transcribing {duration:.2f}s of audio...")
            
            # Skip very short audio
            if duration < 0.5:
                logger.info("⏭️ Audio too short, skipping")
                return ""
            
            # Normalize audio
            audio_normalized = audio_data.astype(np.float32)
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio_normalized,
                language="en",
                fp16=False,
                condition_on_previous_text=False
            )
            
            transcription = result["text"].strip()
            
            if transcription:
                logger.info(f"✅ Transcription: '{transcription}'")
            else:
                logger.info("⚠️ Empty transcription")
                
            return transcription
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def get_llm_response(self, text: str) -> str:
        """Get response from Ollama LLM"""
        try:
            api_url = f"{self.ollama_base_url}/chat/completions"
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful voice assistant. Give brief, natural answers in 1-2 sentences suitable for speaking aloud."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            logger.info("🤖 Querying LLM...")
            response = requests.post(api_url, json=payload, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    answer = data['choices'][0]['message']['content'].strip()
                    logger.info(f"💬 LLM Response: {answer}")
                    return answer
            
            logger.error(f"❌ LLM request failed: {response.status_code}")
            return "I'm having trouble generating a response right now."
            
        except Exception as e:
            logger.error(f"❌ LLM error: {e}")
            return f"Error communicating with AI: {str(e)}"


class AudioProcessor:
    """Handles audio stream processing with proper buffering"""
    
    def __init__(self, agent: VoiceAgent, room: rtc.Room, participant_identity: str):
        self.agent = agent
        self.room = room
        self.participant_identity = participant_identity
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=500)  # Keep last 500 frames (~10 seconds)
        self.is_speaking = False
        self.silence_frames = 0
        self.voice_frames = 0
        
        # Thresholds
        self.SILENCE_THRESHOLD = 60  # Frames of silence to trigger processing
        self.VOICE_THRESHOLD = 800   # Audio level threshold
        self.MIN_VOICE_FRAMES = 20   # Minimum speech frames
        
        # Cooldown to prevent feedback loop
        self.processing = False
        self.last_response_time = 0
        self.COOLDOWN_SECONDS = 3  # Ignore audio for 3 seconds after response
        
    async def process_frame(self, frame: rtc.AudioFrame):
        """Process a single audio frame"""
        try:
            # Check if we're in cooldown period (prevent feedback loop)
            import time
            current_time = time.time()
            if self.processing or (current_time - self.last_response_time) < self.COOLDOWN_SECONDS:
                # Ignore audio during processing or cooldown
                return
            
            # Convert to numpy array
            audio_data = np.frombuffer(frame.data.tobytes(), dtype=np.int16)
            
            # Calculate audio level
            audio_level = np.abs(audio_data.astype(np.float32)).mean()
            
            # Voice activity detection
            if audio_level > self.VOICE_THRESHOLD:
                # Voice detected
                if not self.is_speaking:
                    logger.info(f"🎙️ Voice started (level: {audio_level:.0f})")
                    self.is_speaking = True
                    self.audio_buffer.clear()
                
                self.voice_frames += 1
                self.silence_frames = 0
                self.audio_buffer.append(audio_data)
                
            else:
                # Silence detected
                if self.is_speaking:
                    self.silence_frames += 1
                    self.audio_buffer.append(audio_data)
                    
                    # Check if speech has ended
                    if self.silence_frames >= self.SILENCE_THRESHOLD:
                        if self.voice_frames >= self.MIN_VOICE_FRAMES:
                            logger.info(f"🔄 Speech ended. Processing {self.voice_frames} voice frames...")
                            self.processing = True  # Set flag to ignore new audio
                            await self.process_speech()
                            self.processing = False  # Clear flag after processing
                        else:
                            logger.info(f"⏭️ Speech too short ({self.voice_frames} frames), ignoring")
                        
                        # Reset
                        self.is_speaking = False
                        self.voice_frames = 0
                        self.silence_frames = 0
                        self.audio_buffer.clear()
                        
        except Exception as e:
            logger.error(f"❌ Frame processing error: {e}")
    
    async def process_speech(self):
        """Process accumulated speech audio"""
        try:
            import time
            
            # Concatenate all buffered audio
            audio_combined = np.concatenate(list(self.audio_buffer))
            
            # Normalize to float32 [-1, 1]
            audio_float = audio_combined.astype(np.float32) / 32768.0
            
            # Resample to 16kHz for Whisper if needed
            if self.agent.sample_rate != 16000:
                audio_float = self.resample(audio_float, self.agent.sample_rate, 16000)
            
            # Transcribe
            transcription = self.agent.transcribe(audio_float)
            
            if transcription and len(transcription) > 3:
                # Get LLM response
                response = self.agent.get_llm_response(transcription)
                
                # Send to client via data channel (client will handle TTS via browser)
                message = json.dumps({
                    "type": "conversation",
                    "user": transcription,
                    "assistant": response,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                await self.room.local_participant.publish_data(
                    message.encode('utf-8'),
                    reliable=True
                )
                
                # Update last response time for cooldown
                self.last_response_time = time.time()
                
                logger.info(f"✅ Response sent - Entering {self.COOLDOWN_SECONDS}s cooldown to prevent feedback")
                
        except Exception as e:
            logger.error(f"❌ Speech processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling"""
        if orig_sr == target_sr:
            return audio
        
        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio)


async def main():
    """Main entry point"""
    
    # Configuration
    livekit_url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
    room_name = os.getenv("ROOM_NAME", "voice-room")
    
    logger.info("=" * 70)
    logger.info("🚀 Starting LiveKit Voice Agent V2")
    logger.info("=" * 70)
    logger.info(f"   Server: {livekit_url}")
    logger.info(f"   Room: {room_name}")
    logger.info("=" * 70)
    
    # Create agent
    agent = VoiceAgent()
    
    # Generate token
    token_api = api.AccessToken(livekit_api_key, livekit_api_secret)
    token_api.with_identity("voice-agent-v2").with_name("Voice Assistant").with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
        )
    )
    token = token_api.to_jwt()
    logger.info("🔑 Access token generated")
    
    # Create room
    room = rtc.Room()
    audio_processors = {}
    
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        logger.info(f"📻 Subscribed to {participant.identity}'s {track.kind} track")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # Create audio processor for this participant
            processor = AudioProcessor(agent, room, participant.identity)
            audio_processors[participant.identity] = processor
            
            # Start processing audio stream
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(process_audio_stream(audio_stream, processor))
    
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"👋 {participant.identity} joined the room")
    
    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"👋 {participant.identity} left the room")
        if participant.identity in audio_processors:
            del audio_processors[participant.identity]
    
    @room.on("disconnected")
    def on_disconnected():
        logger.info("❌ Disconnected from room")
    
    try:
        # Connect to room
        await room.connect(livekit_url, token)
        logger.info(f"✅ Connected to room: {room_name}")
        logger.info("🎧 Listening for participants and audio...")
        logger.info("=" * 70)
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Shutting down...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await room.disconnect()
        logger.info("👋 Disconnected")


async def process_audio_stream(audio_stream: rtc.AudioStream, processor: AudioProcessor):
    """Process audio stream frames"""
    try:
        async for frame_event in audio_stream:
            frame = frame_event.frame
            await processor.process_frame(frame)
    except Exception as e:
        logger.error(f"❌ Audio stream error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✅ Agent stopped")
