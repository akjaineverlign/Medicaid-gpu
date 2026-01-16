"""
GPU Voice Handler with External XTTS Server Integration
Connects to your standalone XTTS server via HTTP
"""

import asyncio
import json
import base64
import time
import random
import os
import torch
import numpy as np
from faster_whisper import WhisperModel
from medicaid_voice_agent import CallSession, is_question
import requests
from io import BytesIO
import soundfile as sf

# ==================== CONFIGURATION ====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Voice Handler using device: {DEVICE}")

# Voice Activity Detection Settings
VAD_SILENCE_THRESHOLD = 1.5
VAD_MIN_SPEECH_DURATION = 0.3
INTERRUPTION_COOLDOWN = 0.5

# ==================== CONVERSATIONAL ELEMENTS ====================

THINKING_PHRASES = [
    "Let me see...",
    "One moment...",
    "Hmm, let me check that...",
    "Okay, so...",
    "Alright..."
]

ACKS = [
    "Got it, thank you.",
    "Perfect, thanks for that.",
    "Okay, I have that down.",
    "Great, noted.",
    "Thank you for sharing that."
]

TRANSITIONS = [
    "Alright, moving on...",
    "Okay, next thing...",
    "Great, now let's talk about...",
    "Perfect. So...",
    "Wonderful. Now..."
]

EMOTION_PATTERNS = {
    "uncertain": ["not sure", "don't know", "maybe", "I think", "I guess"],
    "stressed": ["confusing", "confused", "hard", "difficult", "frustrated"],
    "apologetic": ["sorry", "apologize", "my bad"]
}

EMOTION_RESPONSES = {
    "uncertain": [
        "That's completely okay. Take your time.",
        "No worries at all. We can work through this together.",
        "No problem. Let me help clarify."
    ],
    "stressed": [
        "I understand this can be a lot. We'll take it slow.",
        "I hear you. Let's keep this simple.",
        "That's okay. We'll make this as easy as possible."
    ],
    "apologetic": [
        "No need to apologize at all!",
        "You're doing great, no worries.",
        "Not a problem at all."
    ]
}

RAG_TRANSITIONS = [
    "That's a great question. Let me answer that for you...",
    "Good question. So...",
    "I'm glad you asked...",
    "Let me explain that...",
    "Here's what I can tell you about that..."
]

RAG_FOLLOW_UPS = [
    "Does that answer your question?",
    "Does that make sense?",
    "Is there anything else you'd like to know about that?",
    "Hope that helps. Anything else?",
    "Did that clear things up?"
]

INTERRUPTION_ACKS = [
    "Oh, sorry, go ahead.",
    "Yes, what was that?",
    "I'm listening.",
    "Yes?",
    "Go on."
]

MAX_TTS_SECONDS = 8.0

# ==================== HELPER FUNCTIONS ====================

def trim_for_tts(text, max_seconds=MAX_TTS_SECONDS):
    words = text.split()
    max_words = int(max_seconds * 2.5)
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text

def detect_emotion(text: str):
    text = text.lower()
    for emotion, patterns in EMOTION_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return emotion
    return None

def is_interruption(text: str, context: str = ""):
    interruption_signals = [
        "wait", "hold on", "excuse me", "sorry", "actually",
        "but", "what about", "one thing", "quick question"
    ]
    text_lower = text.lower()
    return any(signal in text_lower for signal in interruption_signals)

# ==================== EXTERNAL XTTS CLIENT ====================

class ExternalXTTSClient:
    """
    Client for external XTTS server
    Communicates via HTTP requests
    """
    
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        print(f"🔧 Initializing External XTTS Client...")
        print(f"   Server URL: {self.server_url}")
        
        # Test connectivity
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Connected to XTTS server")
                print(f"   Device: {health_data.get('device')}")
                print(f"   Model: {health_data.get('model')}")
                print(f"   Default Voice: {health_data.get('default_voice', 'None')}")
                self.available = True
            else:
                print(f"⚠️  XTTS server returned status {response.status_code}")
                self.available = False
        except Exception as e:
            print(f"❌ Cannot connect to XTTS server: {e}")
            print(f"   Make sure server is running at {self.server_url}")
            self.available = False
    
    def generate_speech(self, text, language="en", output_format="mulaw"):
        """
        Generate speech via HTTP request
        
        Args:
            text: Text to synthesize
            language: Language code (default: "en")
            output_format: "wav" or "mulaw" (default: "mulaw" for Twilio)
        
        Returns:
            Audio data as bytes (mu-law for Twilio, or WAV)
        """
        if not self.available:
            print("❌ XTTS server not available")
            return None
        
        try:
            print(f"🎤 Generating speech via XTTS server: '{text[:50]}...'")
            start_time = time.time()
            
            # Make request to XTTS server
            response = requests.post(
                f"{self.server_url}/tts",
                json={
                    "text": text,
                    "language": language
                },
                params={
                    "format": output_format  # "mulaw" or "wav"
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ XTTS server error: {response.status_code}")
                return None
            
            elapsed = time.time() - start_time
            print(f"✅ Speech generated in {elapsed:.2f}s")
            
            return response.content
        
        except requests.exceptions.Timeout:
            print("❌ XTTS request timed out")
            return None
        except Exception as e:
            print(f"❌ XTTS error: {e}")
            return None
    
    def test_generation(self):
        """Test speech generation"""
        try:
            print("🧪 Testing XTTS generation...")
            response = requests.get(f"{self.server_url}/test", timeout=30)
            
            if response.status_code == 200:
                print("✅ Test generation successful")
                return True
            else:
                print(f"❌ Test failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Test error: {e}")
            return False

# ==================== CONVERSATION CONTEXT MANAGER ====================

class ConversationContext:
    """Manages full call context for context-aware responses"""
    
    def __init__(self, call_sid, member_id):
        self.call_sid = call_sid
        self.member_id = member_id
        self.history = []
        self.start_time = time.time()
        
    def add_user_message(self, message):
        self.history.append({
            "role": "user",
            "content": message,
            "timestamp": time.time()
        })
        
    def add_agent_message(self, message):
        self.history.append({
            "role": "agent",
            "content": message,
            "timestamp": time.time()
        })
    
    def get_recent_context(self, n=5):
        return self.history[-n*2:] if self.history else []
    
    def format_for_llm(self):
        formatted = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Agent"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def get_conversation_summary(self):
        if not self.history:
            return "No conversation yet."
        
        return {
            "total_exchanges": len(self.history) // 2,
            "duration": time.time() - self.start_time,
            "message_count": len(self.history)
        }

# ==================== GPU VOICE HANDLER ====================

class GPUVoiceHandler:
    """
    Enhanced voice handler with external XTTS server
    All TTS processing delegated to standalone server
    """

    def __init__(self, xtts_server_url):
        print("🔧 Initializing GPU Voice Handler...")
        
        # Initialize Faster Whisper (GPU-accelerated STT)
        print("📥 Loading Faster Whisper model...")
        self.whisper = WhisperModel(
            "large-v3",
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8",
            vad_filter=True
        )
        print("✅ Whisper loaded with VAD")
        
        # Initialize External XTTS Client
        print("🎤 Connecting to external XTTS server...")
        self.tts = ExternalXTTSClient(server_url=xtts_server_url)
        print("✅ XTTS client initialized")
        
        # Active connections
        self.active_connections = {}
        self.conversation_contexts = {}
        
        # Audio buffers
        self.audio_buffers = {}
        
        # VAD state
        self.vad_state = {}

    async def handle_call(self, websocket, call_sid, session: CallSession):
        """Main call handler with VAD and interruption handling"""
        
        print(f"📞 Starting call: {call_sid}")
        
        # Initialize conversation context
        context = ConversationContext(call_sid, session.member_id)
        self.conversation_contexts[call_sid] = context
        
        # Initialize audio buffer
        self.audio_buffers[call_sid] = bytearray()
        
        # Initialize VAD state
        self.vad_state[call_sid] = {
            "is_speaking": False,
            "last_speech_time": 0,
            "last_silence_time": 0,
            "speech_start_time": 0,
            "user_finished_speaking": False
        }
        
        # Store connection info
        self.active_connections[call_sid] = {
            "websocket": websocket,
            "session": session,
            "agent_speaking": False,
            "context": context
        }

        try:
            # Start with first question
            first_question = session.ask_question()
            context.add_agent_message(first_question)
            await self.speak(call_sid, first_question)

            # Main conversation loop
            async for message in websocket:
                data = json.loads(message)

                if data["event"] == "media":
                    # Accumulate audio data
                    audio_payload = base64.b64decode(data["media"]["payload"])
                    self.audio_buffers[call_sid].extend(audio_payload)
                    
                    # Update VAD state
                    vad = self.vad_state[call_sid]
                    current_time = time.time()
                    
                    # Simple energy-based detection
                    audio_energy = np.frombuffer(audio_payload, dtype=np.uint8).std()
                    
                    if audio_energy > 10:
                        if not vad["is_speaking"]:
                            vad["speech_start_time"] = current_time
                            vad["is_speaking"] = True
                            print(f"🎤 [{call_sid}] User started speaking")
                            
                            # Check for interruption
                            conn = self.active_connections.get(call_sid)
                            if conn and conn.get("agent_speaking"):
                                print(f"✋ [{call_sid}] User interrupted agent!")
                                conn["agent_speaking"] = False
                                
                                await self.speak(
                                    call_sid,
                                    random.choice(INTERRUPTION_ACKS),
                                    low_volume=True
                                )
                                await asyncio.sleep(INTERRUPTION_COOLDOWN)
                        
                        vad["last_speech_time"] = current_time
                        vad["user_finished_speaking"] = False
                    else:
                        vad["last_silence_time"] = current_time
                    
                    # Check if user finished
                    if vad["is_speaking"]:
                        silence_duration = current_time - vad["last_speech_time"]
                        speech_duration = current_time - vad["speech_start_time"]
                        
                        if (silence_duration >= VAD_SILENCE_THRESHOLD and 
                            speech_duration >= VAD_MIN_SPEECH_DURATION):
                            vad["is_speaking"] = False
                            vad["user_finished_speaking"] = True
                            print(f"✅ [{call_sid}] User finished speaking")
                    
                    # Process audio when buffer is large enough
                    if len(self.audio_buffers[call_sid]) > 16000 * 2:
                        if vad["user_finished_speaking"]:
                            await self.process_audio_chunk(call_sid)
                            vad["user_finished_speaking"] = False

                elif data["event"] == "stop":
                    print(f"📞 Call ended: {call_sid}")
                    break

        finally:
            # Cleanup
            self.active_connections.pop(call_sid, None)
            self.audio_buffers.pop(call_sid, None)
            self.vad_state.pop(call_sid, None)
            self.save_conversation_context(call_sid)
            print(f"🧹 Cleaned up connection: {call_sid}")

    async def process_audio_chunk(self, call_sid):
        """Process audio using Faster Whisper"""
        
        audio_data = bytes(self.audio_buffers[call_sid])
        self.audio_buffers[call_sid] = bytearray()
        
        if len(audio_data) < 1600:
            return
        
        try:
            # Convert to float32
            audio_np = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
            audio_np = (audio_np - 128.0) / 128.0
            
            # Transcribe
            segments, info = self.whisper.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                vad_filter=True
            )
            
            transcript = " ".join([segment.text for segment in segments]).strip()
            
            if transcript:
                print(f"👤 [{call_sid}] User: {transcript}")
                await self.process_user_input(call_sid, transcript)
        
        except Exception as e:
            print(f"❌ Error processing audio: {e}")

    async def process_user_input(self, call_sid, transcript):
        """Process user's speech"""
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        session = conn["session"]
        context = conn["context"]
        
        context.add_user_message(transcript)
        print(f"👤 [{call_sid}] User: {transcript}")

        # Emotion detection
        emotion = detect_emotion(transcript)
        if emotion and emotion in EMOTION_RESPONSES:
            empathy_response = random.choice(EMOTION_RESPONSES[emotion])
            context.add_agent_message(empathy_response)
            await self.speak(call_sid, empathy_response, low_volume=True)
            await asyncio.sleep(0.4)

        # Check for question
        if is_question(transcript):
            print(f"❓ [{call_sid}] Question detected, using GPU RAG...")
            
            transition = random.choice(RAG_TRANSITIONS)
            context.add_agent_message(transition)
            await self.speak(call_sid, transition, low_volume=True)
            await asyncio.sleep(0.3)
            
            rag_answer = session.answer_user_question(transcript)
            
            context.add_agent_message(rag_answer)
            await self.speak(call_sid, rag_answer)
            await asyncio.sleep(0.5)
            
            follow_up = random.choice(RAG_FOLLOW_UPS)
            context.add_agent_message(follow_up)
            await self.speak(call_sid, follow_up)
            
            return

        # Normal flow
        if random.random() < 0.3:
            thinking = random.choice(THINKING_PHRASES)
            context.add_agent_message(thinking)
            await self.speak(call_sid, thinking, low_volume=True)
            await asyncio.sleep(0.2)

        agent_response, should_advance = session.handle_response(transcript)
        
        print(f"🤖 [{call_sid}] Agent: {agent_response}")
        context.add_agent_message(agent_response)

        await self.speak(call_sid, agent_response)

        if should_advance:
            await asyncio.sleep(0.4)
            
            ack = random.choice(ACKS)
            context.add_agent_message(ack)
            await self.speak(call_sid, ack, low_volume=True)
            await asyncio.sleep(0.3)
            
            transition = random.choice(TRANSITIONS)
            context.add_agent_message(transition)
            await self.speak(call_sid, transition, low_volume=True)
            await asyncio.sleep(0.4)

            session.advance_step()
            current_step = session.get_current_step()

            if current_step is None:
                await self.end_call(call_sid)
            elif current_step == "close":
                final_message = session.ask_question()
                context.add_agent_message(final_message)
                await self.speak(call_sid, final_message)
                session.form["status"] = "REDETERMINATION_COMPLETE"
                await self.end_call(call_sid)
            else:
                if session.needs_question():
                    next_question = session.ask_question()
                    context.add_agent_message(next_question)
                    await self.speak(call_sid, next_question)

    async def speak(self, call_sid, text, low_volume=False):
        """Generate and stream speech using external XTTS server"""
        
        text = trim_for_tts(text)
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        try:
            if not low_volume:
                conn["agent_speaking"] = True

            # Generate speech via external XTTS server
            # Request mu-law format directly (optimized for Twilio)
            audio_data = self.tts.generate_speech(
                text=text,
                language="en",
                output_format="mulaw"  # Get mu-law directly from server
            )
            
            if audio_data is None:
                print(f"❌ Failed to generate speech")
                conn["agent_speaking"] = False
                return

            # Audio is already in mu-law 8kHz format from server
            mulaw = audio_data

            # Send in chunks
            CHUNK_SIZE = 160  # 20ms at 8kHz
            for i in range(0, len(mulaw), CHUNK_SIZE):
                chunk = mulaw[i:i + CHUNK_SIZE]
                audio_base64 = base64.b64encode(chunk).decode()
                
                message = {
                    "event": "media",
                    "streamSid": call_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                await conn["websocket"].send(json.dumps(message))
                await asyncio.sleep(0.02)  # 20ms delay

            if not low_volume:
                conn["agent_speaking"] = False

        except Exception as e:
            print(f"❌ TTS error: {e}")
            conn["agent_speaking"] = False

    def save_conversation_context(self, call_sid):
        """Save conversation to file"""
        
        context = self.conversation_contexts.get(call_sid)
        if not context:
            return
        
        os.makedirs("call_logs", exist_ok=True)
        
        context_data = {
            "call_sid": call_sid,
            "member_id": context.member_id,
            "start_time": context.start_time,
            "end_time": time.time(),
            "duration": time.time() - context.start_time,
            "full_history": context.history,
            "summary": context.get_conversation_summary()
        }
        
        with open(f"call_logs/{call_sid}_context.json", "w") as f:
            json.dump(context_data, f, indent=2)
        
        print(f"💾 Conversation context saved")

    async def end_call(self, call_sid):
        """End call gracefully"""
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        session = conn["session"]

        os.makedirs("call_logs", exist_ok=True)
        log_path = f"call_logs/{call_sid}.json"
        
        with open(log_path, "w") as f:
            json.dump(session.form, f, indent=2)

        print(f"💾 Call log saved: {log_path}")
        self.save_conversation_context(call_sid)

        if conn["websocket"]:
            await conn["websocket"].close()

        print(f"✅ Call ended: {call_sid}")

# ==================== SINGLETON ====================

_voice_handler = None

def get_voice_handler(xtts_server_url=None):
    """Get or create voice handler"""
    global _voice_handler
    if _voice_handler is None:
        if xtts_server_url is None:
            xtts_server_url = os.getenv('XTTS_SERVER_URL', 'http://localhost:8000')
        _voice_handler = GPUVoiceHandler(xtts_server_url=xtts_server_url)
    return _voice_handler