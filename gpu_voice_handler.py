"""
GPU Voice Handler with SocketIO Support
Works with Flask-SocketIO instead of raw WebSocket
"""

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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Voice Handler using device: {DEVICE}")

# Voice Activity Detection Settings
VAD_SILENCE_THRESHOLD = 1.5
VAD_MIN_SPEECH_DURATION = 0.3
INTERRUPTION_COOLDOWN = 0.5


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
    """Client for external XTTS server"""
    
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        print(f"🔧 Initializing External XTTS Client...")
        print(f"   Server URL: {self.server_url}")
        
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Connected to XTTS server")
                print(f"   Device: {health_data.get('device')}")
                self.available = True
            else:
                print(f"⚠️  XTTS server returned status {response.status_code}")
                self.available = False
        except Exception as e:
            print(f"❌ Cannot connect to XTTS server: {e}")
            self.available = False
    
    def generate_speech(self, text, language="en", output_format="mulaw"):
        """Generate speech via HTTP request"""
        if not self.available:
            return None
        
        try:
            print(f"🎤 Generating speech: '{text[:50]}...'")
            start_time = time.time()
            
            response = requests.post(
                f"{self.server_url}/tts",
                json={"text": text, "language": language},
                params={"format": output_format},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ XTTS error: {response.status_code}")
                return None
            
            elapsed = time.time() - start_time
            print(f"✅ Speech generated in {elapsed:.2f}s")
            
            return response.content
        
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None

# ==================== CONVERSATION CONTEXT ====================

class ConversationContext:
    """Manages conversation context"""
    
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
    
    def format_for_llm(self):
        formatted = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Agent"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

# ==================== GPU VOICE HANDLER ====================

class GPUVoiceHandler:
    """Voice handler with Flask-SocketIO support"""

    def __init__(self, xtts_server_url):
        print("🔧 Initializing GPU Voice Handler...")
        
        # Initialize Faster Whisper
        print("📥 Loading Faster Whisper model...")
        self.whisper = WhisperModel(
            "large-v3",
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8",
            vad_filter=True
        )
        print("✅ Whisper loaded")
        
        # Initialize XTTS Client
        self.tts = ExternalXTTSClient(server_url=xtts_server_url)
        
        # Active connections
        self.active_connections = {}
        self.conversation_contexts = {}
        self.audio_buffers = {}
        self.vad_state = {}

    async def handle_call_socketio(self, socket_id, call_sid, session: CallSession, socketio):
        """Main call handler for SocketIO"""
        
        print(f"📞 Starting call: {call_sid}")
        
        # Initialize context
        context = ConversationContext(call_sid, session.member_id)
        self.conversation_contexts[call_sid] = context
        
        # Initialize buffers
        self.audio_buffers[call_sid] = bytearray()
        
        # Initialize VAD
        self.vad_state[call_sid] = {
            "is_speaking": False,
            "last_speech_time": 0,
            "last_silence_time": 0,
            "speech_start_time": 0,
            "user_finished_speaking": False
        }
        
        # Store connection
        self.active_connections[call_sid] = {
            "socket_id": socket_id,
            "session": session,
            "agent_speaking": False,
            "context": context,
            "socketio": socketio
        }

        try:
            # Start with first question
            first_question = session.ask_question()
            context.add_agent_message(first_question)
            await self.speak_socketio(call_sid, first_question, socketio)
            
            # Note: Actual message processing happens in handle_message
            # This coroutine just sets up the call
            
        except Exception as e:
            print(f"❌ Error in call handler: {e}")
            import traceback
            traceback.print_exc()

    def process_media_message(self, call_sid, payload):
        """Process incoming audio (called from SocketIO handler)"""
        
        # Decode audio
        audio_payload = base64.b64decode(payload)
        self.audio_buffers[call_sid].extend(audio_payload)
        
        # Update VAD
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
                    print(f"✋ [{call_sid}] User interrupted!")
                    conn["agent_speaking"] = False
            
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
        
        # Process when ready
        if len(self.audio_buffers[call_sid]) > 16000 * 2:
            if vad["user_finished_speaking"]:
                self.process_audio_chunk_sync(call_sid)
                vad["user_finished_speaking"] = False

    def process_audio_chunk_sync(self, call_sid):
        """Process audio synchronously (for SocketIO)"""
        
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
                self.process_user_input_sync(call_sid, transcript)
        
        except Exception as e:
            print(f"❌ Error processing audio: {e}")

    def process_user_input_sync(self, call_sid, transcript):
        """Process user input synchronously"""
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        session = conn["session"]
        context = conn["context"]
        socketio = conn["socketio"]
        
        context.add_user_message(transcript)

        # Emotion detection
        emotion = detect_emotion(transcript)
        if emotion and emotion in EMOTION_RESPONSES:
            empathy_response = random.choice(EMOTION_RESPONSES[emotion])
            context.add_agent_message(empathy_response)
            self.speak_socketio_sync(call_sid, empathy_response, socketio, low_volume=True)
            time.sleep(0.4)

        # Check for question
        if is_question(transcript):
            print(f"❓ [{call_sid}] Question detected")
            
            transition = random.choice(RAG_TRANSITIONS)
            context.add_agent_message(transition)
            self.speak_socketio_sync(call_sid, transition, socketio, low_volume=True)
            time.sleep(0.3)
            
            rag_answer = session.answer_user_question(transcript)
            context.add_agent_message(rag_answer)
            self.speak_socketio_sync(call_sid, rag_answer, socketio)
            time.sleep(0.5)
            
            follow_up = random.choice(RAG_FOLLOW_UPS)
            context.add_agent_message(follow_up)
            self.speak_socketio_sync(call_sid, follow_up, socketio)
            time.sleep(0.3)
            if session.needs_question():
                    next_question = session.ask_question()
                    context.add_agent_message(next_question)
                    self.speak_socketio_sync(call_sid, next_question, socketio)
            
            return

        # Normal flow
        if random.random() < 0.3:
            thinking = random.choice(THINKING_PHRASES)
            context.add_agent_message(thinking)
            self.speak_socketio_sync(call_sid, thinking, socketio, low_volume=True)
            time.sleep(0.2)

        agent_response, should_advance = session.handle_response(transcript)
        
        print(f"🤖 [{call_sid}] Agent: {agent_response}")
        context.add_agent_message(agent_response)

        self.speak_socketio_sync(call_sid, agent_response, socketio)

        if should_advance:
            time.sleep(0.4)
            
            ack = random.choice(ACKS)
            context.add_agent_message(ack)
            self.speak_socketio_sync(call_sid, ack, socketio, low_volume=True)
            time.sleep(0.3)
            
            session.advance_step()
            current_step = session.get_current_step()

            if current_step is None or current_step == "close":
                final_message = session.ask_question() if current_step == "close" else "Thank you for your time!"
                context.add_agent_message(final_message)
                self.speak_socketio_sync(call_sid, final_message, socketio)
                session.form["status"] = "REDETERMINATION_COMPLETE"
            else:
                if session.needs_question():
                    next_question = session.ask_question()
                    context.add_agent_message(next_question)
                    self.speak_socketio_sync(call_sid, next_question, socketio)

    async def speak_socketio(self, call_sid, text, socketio, low_volume=False):
        """Async wrapper for speaking"""
        self.speak_socketio_sync(call_sid, text, socketio, low_volume)

    def speak_socketio_sync(self, call_sid, text, socketio, low_volume=False):
        """Generate and stream speech using SocketIO"""
        
        text = trim_for_tts(text)
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        try:
            if not low_volume:
                conn["agent_speaking"] = True

            # Generate speech
            audio_data = self.tts.generate_speech(
                text=text,
                language="en",
                output_format="mulaw"
            )
            
            if audio_data is None:
                print(f"❌ Failed to generate speech")
                conn["agent_speaking"] = False
                return

            # Send in chunks via SocketIO
            CHUNK_SIZE = 160  # 20ms at 8kHz
            for i in range(0, len(audio_data), CHUNK_SIZE):
                chunk = audio_data[i:i + CHUNK_SIZE]
                audio_base64 = base64.b64encode(chunk).decode()
                
                message = {
                    "event": "media",
                    "streamSid": call_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                # Emit to specific socket
                socketio.emit('message', json.dumps(message), namespace='/stream', room=conn["socket_id"])
                time.sleep(0.02)

            if not low_volume:
                conn["agent_speaking"] = False

        except Exception as e:
            print(f"❌ TTS error: {e}")
            conn["agent_speaking"] = False



_voice_handler = None

def get_voice_handler(xtts_server_url=None):
    """Get or create voice handler"""
    global _voice_handler
    if _voice_handler is None:
        if xtts_server_url is None:
            xtts_server_url = os.getenv('XTTS_SERVER_URL', 'http://localhost:8000')
        _voice_handler = GPUVoiceHandler(xtts_server_url=xtts_server_url)
    return _voice_handler