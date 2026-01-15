"""
GPU Voice Handler with Remote XTTS Server (Mac)
- Connects to external XTTS server for voice cloning
- Smart interruption handling
- Waits for user to finish speaking before continuing
- Voice Activity Detection (VAD) for natural pauses
"""

import asyncio
import json
import base64
import time
import random
import os
import torch
import numpy as np
import requests
from faster_whisper import WhisperModel
from medicaid_voice_agent import CallSession, is_question


# ==================== CONFIGURATION ====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# Remote XTTS Server
XTTS_REMOTE_URL = "https://80ed611dc031.ngrok-free.app"
print(f"🎤 XTTS Server: {XTTS_REMOTE_URL}")

# Voice Activity Detection Settings
VAD_SILENCE_THRESHOLD = 1.5  # seconds of silence before considering user finished
VAD_MIN_SPEECH_DURATION = 0.3  # minimum speech duration to count as user input
INTERRUPTION_COOLDOWN = 0.5  # cooldown after interruption before agent continues

# ==================== CONVERSATIONAL ELEMENTS ====================

THINKING_PHRASES = [
    "Let me see...",
    "One moment...",
    "Hmm, let me check that...",
    "Okay, so...",
    "Alright..."
]

BACKCHANNELS = ["mm-hmm", "I see", "okay", "right", "got it"]

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

SILENCE_PROMPTS = [
    "I'm here whenever you're ready.",
    "Take your time.",
    "No rush.",
    "I'm listening."
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
    
    def get_full_context(self):
        return self.history
    
    def format_for_llm(self):
        formatted = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Agent"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def get_conversation_summary(self):
        if not self.history:
            return "No conversation yet."
        
        user_messages = [m["content"] for m in self.history if m["role"] == "user"]
        agent_messages = [m["content"] for m in self.history if m["role"] == "agent"]
        
        return {
            "total_exchanges": len(self.history) // 2,
            "duration": time.time() - self.start_time,
            "user_messages": len(user_messages),
            "agent_messages": len(agent_messages)
        }


# ==================== REMOTE XTTS CLIENT ====================

class RemoteXTTSClient:
    """Client for remote XTTS server running on Mac"""
    
    def __init__(self, base_url, voice_sample_path=None):
        self.base_url = base_url.rstrip('/')
        self.voice_sample_path = voice_sample_path
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to XTTS server: {self.base_url}")
            else:
                print(f"⚠️ XTTS server responded with status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not connect to XTTS server: {e}")
            print("   Make sure your Mac XTTS server is running!")
    
    def tts(self, text, language="en"):
        """
        Generate speech using remote XTTS server
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Audio data as bytes
        """
        try:
            # Prepare request
            data = {
                "text": text,
                "language": language
            }
            
            # If voice sample is provided, send it
            files = None
            if self.voice_sample_path and os.path.exists(self.voice_sample_path):
                with open(self.voice_sample_path, 'rb') as f:
                    files = {'speaker_wav': f}
                    response = requests.post(
                        f"{self.base_url}/tts",
                        data=data,
                        files=files,
                        timeout=30
                    )
            else:
                # No voice sample, use default voice
                response = requests.post(
                    f"{self.base_url}/tts",
                    json=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"⚠️ TTS request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None


# ==================== GPU VOICE HANDLER ====================

class GPUVoiceHandler:
    """
    Enhanced voice handler with:
    - Remote XTTS for voice cloning (Mac server)
    - Smart interruption handling
    - VAD for detecting when user finishes speaking
    - Natural conversation flow
    """

    def __init__(self, voice_sample_path=None):
        print("🔧 Initializing GPU Voice Handler...")
        
        # Initialize Faster Whisper (GPU-accelerated STT)
        print("📥 Loading Faster Whisper model...")
        self.whisper = WhisperModel(
            "large-v3",
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8",
            vad_filter=True  # Enable Voice Activity Detection
        )
        print("✅ Whisper loaded with VAD")
        
        # Initialize Remote XTTS Client
        print("🎤 Connecting to remote XTTS server...")
        self.tts = RemoteXTTSClient(
            base_url=XTTS_REMOTE_URL,
            voice_sample_path=voice_sample_path
        )
        print("✅ XTTS client initialized")
        
        # Active connections
        self.active_connections = {}
        self.conversation_contexts = {}
        
        # Audio buffers for streaming
        self.audio_buffers = {}
        
        # Voice Activity Detection state
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
            "user_finished_speaking": False,
            "agent_can_speak": True
        }
        
        # Conversation state tracking
        agent_speaking = False
        agent_was_interrupted = False
        pending_agent_message = None
        
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
                    
                    # Update VAD state - user is providing audio
                    vad = self.vad_state[call_sid]
                    current_time = time.time()
                    
                    # Check if user is speaking (simple energy-based detection)
                    audio_energy = np.frombuffer(audio_payload, dtype=np.uint8).std()
                    
                    if audio_energy > 10:  # Threshold for speech detection
                        if not vad["is_speaking"]:
                            vad["speech_start_time"] = current_time
                            vad["is_speaking"] = True
                            print(f"🎤 [{call_sid}] User started speaking")
                            
                            # ========== INTERRUPTION DETECTION ==========
                            conn = self.active_connections.get(call_sid)
                            if conn and conn.get("agent_speaking"):
                                print(f"✋ [{call_sid}] User interrupted agent!")
                                conn["agent_speaking"] = False
                                agent_was_interrupted = True
                                
                                # Acknowledge interruption
                                await self.speak(
                                    call_sid,
                                    random.choice(INTERRUPTION_ACKS),
                                    priority=True,
                                    low_volume=True
                                )
                                await asyncio.sleep(INTERRUPTION_COOLDOWN)
                        
                        vad["last_speech_time"] = current_time
                        vad["user_finished_speaking"] = False
                    else:
                        # Silence detected
                        vad["last_silence_time"] = current_time
                    
                    # Check if user finished speaking
                    if vad["is_speaking"]:
                        silence_duration = current_time - vad["last_speech_time"]
                        speech_duration = current_time - vad["speech_start_time"]
                        
                        if (silence_duration >= VAD_SILENCE_THRESHOLD and 
                            speech_duration >= VAD_MIN_SPEECH_DURATION):
                            vad["is_speaking"] = False
                            vad["user_finished_speaking"] = True
                            print(f"✅ [{call_sid}] User finished speaking (silence: {silence_duration:.1f}s)")
                    
                    # Process audio chunk when buffer is large enough
                    if len(self.audio_buffers[call_sid]) > 16000 * 2:  # ~2 seconds
                        # Only process if user finished speaking
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
            
            # Save conversation context
            self.save_conversation_context(call_sid)
            
            print(f"🧹 Cleaned up connection: {call_sid}")


    async def process_audio_chunk(self, call_sid):
        """Process accumulated audio using Faster Whisper (GPU)"""
        
        audio_data = bytes(self.audio_buffers[call_sid])
        self.audio_buffers[call_sid] = bytearray()
        
        if len(audio_data) < 1600:  # Too short
            return
        
        try:
            # Convert mu-law to float32 for Whisper
            audio_np = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
            audio_np = (audio_np - 128.0) / 128.0  # Normalize
            
            # Run Whisper inference with VAD
            segments, info = self.whisper.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500
                }
            )
            
            # Get transcript
            transcript = " ".join([segment.text for segment in segments]).strip()
            
            if transcript:
                print(f"👤 [{call_sid}] User: {transcript}")
                await self.process_user_input(call_sid, transcript)
        
        except Exception as e:
            print(f"❌ Error processing audio: {e}")


    async def process_user_input(self, call_sid, transcript):
        """Process user's speech with full conversation context"""
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        session = conn["session"]
        context = conn["context"]
        
        # Add to conversation history
        context.add_user_message(transcript)
        
        print(f"👤 [{call_sid}] User: {transcript}")

        # ========== EMOTION DETECTION & EMPATHY ==========
        emotion = detect_emotion(transcript)
        if emotion and emotion in EMOTION_RESPONSES:
            empathy_response = random.choice(EMOTION_RESPONSES[emotion])
            context.add_agent_message(empathy_response)
            await self.speak(call_sid, empathy_response, low_volume=True)
            await asyncio.sleep(0.4)

        # ========== CHECK IF IT'S A QUESTION (RAG NEEDED) ==========
        if is_question(transcript):
            print(f"❓ [{call_sid}] Question detected, using GPU RAG with full context...")
            
            # Natural transition to RAG answer
            transition = random.choice(RAG_TRANSITIONS)
            context.add_agent_message(transition)
            await self.speak(call_sid, transition, low_volume=True)
            await asyncio.sleep(0.3)
            
            # Get RAG answer using session's method
            rag_answer = session.answer_user_question(transcript)
            
            # Deliver answer
            context.add_agent_message(rag_answer)
            await self.speak(call_sid, rag_answer)
            await asyncio.sleep(0.5)
            
            # Ask follow-up question after RAG
            follow_up = random.choice(RAG_FOLLOW_UPS)
            context.add_agent_message(follow_up)
            await self.speak(call_sid, follow_up)
            
            return

        # ========== NORMAL CONVERSATION FLOW ==========
        if random.random() < 0.3:
            thinking = random.choice(THINKING_PHRASES)
            context.add_agent_message(thinking)
            await self.speak(call_sid, thinking, low_volume=True)
            await asyncio.sleep(0.2)

        # Process response through session logic
        agent_response, should_advance = session.handle_response(transcript)
        
        print(f"🤖 [{call_sid}] Agent: {agent_response}")
        
        # Add to context
        context.add_agent_message(agent_response)

        # Deliver agent's response
        await self.speak(call_sid, agent_response)

        # ========== HANDLE STEP ADVANCEMENT ==========
        if should_advance:
            await asyncio.sleep(0.4)
            
            # Acknowledge before moving on
            ack = random.choice(ACKS)
            context.add_agent_message(ack)
            await self.speak(call_sid, ack, low_volume=True)
            await asyncio.sleep(0.3)
            
            # Natural transition
            transition = random.choice(TRANSITIONS)
            context.add_agent_message(transition)
            await self.speak(call_sid, transition, low_volume=True)
            await asyncio.sleep(0.4)

            # Advance to next step
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


    async def speak(self, call_sid, text, low_volume=False, priority=False):
        """
        Convert text to speech using Remote XTTS server
        """
        
        text = trim_for_tts(text)
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        try:
            # Mark agent as speaking (unless it's a low volume backchannel)
            if not low_volume:
                conn["agent_speaking"] = True

            # Generate speech using remote XTTS server
            print(f"🎤 Generating speech via remote XTTS: '{text[:50]}...'")
            start_time = time.time()
            
            audio_bytes = self.tts.tts(text=text, language="en")
            
            if audio_bytes is None:
                print(f"❌ Failed to generate speech")
                conn["agent_speaking"] = False
                return
            
            elapsed = time.time() - start_time
            print(f"✅ Speech generated in {elapsed:.2f}s ({len(audio_bytes)} bytes)")

            # Convert audio to format suitable for Twilio (mu-law, 8kHz)
            # For now, we'll send the audio as-is and handle conversion on XTTS server
            
            # Send to Twilio via WebSocket
            if conn["websocket"]:
                audio_base64 = base64.b64encode(audio_bytes).decode()
                
                message = {
                    "event": "media",
                    "streamSid": call_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                await conn["websocket"].send(json.dumps(message))

            # Estimate speech duration and wait
            # Assuming 22050 Hz sample rate, 16-bit audio
            duration = len(audio_bytes) / (22050 * 2)
            await asyncio.sleep(duration)

            # Mark agent as done speaking
            if not low_volume:
                conn["agent_speaking"] = False

        except Exception as e:
            print(f"❌ TTS error: {e}")
            conn["agent_speaking"] = False


    def save_conversation_context(self, call_sid):
        """Save full conversation context to file"""
        
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
            "full_history": context.get_full_context(),
            "summary": context.get_conversation_summary()
        }
        
        with open(f"call_logs/{call_sid}_context.json", "w") as f:
            json.dump(context_data, f, indent=2)
        
        print(f"💾 Conversation context saved: call_logs/{call_sid}_context.json")


    async def end_call(self, call_sid):
        """End call gracefully and save all data"""
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        session = conn["session"]

        # Save call log
        os.makedirs("call_logs", exist_ok=True)
        log_path = f"call_logs/{call_sid}.json"
        
        with open(log_path, "w") as f:
            json.dump(session.form, f, indent=2)

        print(f"💾 Call log saved: {log_path}")
        
        # Save conversation context
        self.save_conversation_context(call_sid)

        # Close WebSocket
        if conn["websocket"]:
            await conn["websocket"].close()

        print(f"✅ Call ended successfully: {call_sid}")


# ==================== SINGLETON INSTANCE ====================

_voice_handler = None

def get_voice_handler():
    """Get or create the global voice handler instance"""
    global _voice_handler
    if _voice_handler is None:
        voice_sample = os.getenv("VOICE_SAMPLE_PATH")
        _voice_handler = GPUVoiceHandler(voice_sample_path=voice_sample)
    return _voice_handler

# import asyncio
# import json
# import base64
# import time
# import random
# import os
# import torch
# import numpy as np
# from faster_whisper import WhisperModel
# from medicaid_voice_agent import CallSession, is_question
# import requests
# import soundfile as sf
# from io import BytesIO
# import uuid

# # ==================== XTTS CONFIG ====================

# XTTS_REMOTE_URL = "https://80d055e25e88.ngrok-free.app/"

# def synthesize_xtts_remote(text: str) -> np.ndarray:
#     resp = requests.post(
#         XTTS_REMOTE_URL,
#         json={"text": text, "language": "en"},
#         timeout=60
#     )
#     resp.raise_for_status()
#     wav, _ = sf.read(BytesIO(resp.content), dtype="float32")
#     return wav



# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🚀 Using device: {DEVICE}")



# THINKING_PHRASES = [
#     "Let me see...", "One moment...", "Hmm, let me check that...",
#     "Okay, so...", "Alright..."
# ]

# BACKCHANNELS = ["mm-hmm", "I see", "okay", "right", "got it"]

# ACKS = [
#     "Got it, thank you.", "Perfect, thanks for that.",
#     "Okay, I have that down.", "Great, noted.", "Thank you for sharing that."
# ]

# TRANSITIONS = [
#     "Alright, moving on...", "Okay, next thing...",
#     "Great, now let's talk about...", "Perfect. So...",
#     "Wonderful. Now..."
# ]

# INTERRUPTION_ACKS = [
#     "Oh, sorry, go ahead.", "Yes, what was that?",
#     "I'm listening.", "Go on."
# ]

# SILENCE_PROMPTS = [
#     "I'm here whenever you're ready.",
#     "Take your time.",
#     "No rush.",
#     "I'm listening."
# ]

# MAX_TTS_SECONDS = 8.0


# # ==================== HELPERS ====================

# def trim_for_tts(text, max_seconds=MAX_TTS_SECONDS):
#     words = text.split()
#     max_words = int(max_seconds * 2.5)
#     return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

# def is_interruption(text: str):
#     signals = [
#         "wait", "hold on", "excuse me", "sorry",
#         "actually", "but", "what about", "one thing"
#     ]
#     return any(s in text.lower() for s in signals)


# # ==================== VOICE HANDLER ====================

# class GPUVoiceHandler:

#     def __init__(self):
#         print("🔧 Initializing GPU Voice Handler...")

#         self.whisper = WhisperModel(
#             "large-v3",
#             device=DEVICE,
#             compute_type="float16" if DEVICE == "cuda" else "int8"
#         )

#         self.active_connections = {}
#         self.audio_buffers = {}
#         self.playback_tokens = {}

#     # ==================== MAIN CALL ====================

#     async def handle_call(self, websocket, call_sid, session: CallSession):

#         self.audio_buffers[call_sid] = bytearray()

#         self.active_connections[call_sid] = {
#             "websocket": websocket,
#             "session": session,
#             "agent_speaking": False,
#             "last_user_text": ""
#         }

#         try:
#             first_question = session.ask_question()
#             await self.speak(call_sid, first_question)

#             async for message in websocket:
#                 data = json.loads(message)

#                 if data["event"] == "media":
#                     audio = base64.b64decode(data["media"]["payload"])
#                     self.audio_buffers[call_sid].extend(audio)

#                     if len(self.audio_buffers[call_sid]) > 16000 * 2:
#                         await self.process_audio(call_sid)

#                 elif data["event"] == "stop":
#                     break

#                 # ========== INTERRUPTION ==========
#                 conn = self.active_connections.get(call_sid)
#                 if not conn:
#                     continue

#                 interim = conn.get("last_user_text", "")

#                 if conn["agent_speaking"] and interim and is_interruption(interim):
#                     print("🛑 INTERRUPTION — stopping agent audio")

#                     self.playback_tokens[call_sid] = None
#                     conn["agent_speaking"] = False

#                     await conn["websocket"].send(json.dumps({
#                         "event": "clear",
#                         "streamSid": call_sid
#                     }))

#                     await asyncio.sleep(0.05)
#                     await self.speak(call_sid, random.choice(INTERRUPTION_ACKS), priority=True)

#         finally:
#             self.active_connections.pop(call_sid, None)
#             self.audio_buffers.pop(call_sid, None)
#             print(f"🧹 Cleaned up call: {call_sid}")

#     # ==================== AUDIO PROCESS ====================

#     async def process_audio(self, call_sid):
#         conn = self.active_connections.get(call_sid)
#         if not conn:
#             return

#         audio = bytes(self.audio_buffers[call_sid])
#         self.audio_buffers[call_sid] = bytearray()

#         segments, _ = self.whisper.transcribe(
#             np.frombuffer(audio, dtype=np.uint8),
#             beam_size=5,
#             language="en",
#             vad_filter=True
#         )

#         text = " ".join(s.text for s in segments).strip()
#         if text:
#             conn["last_user_text"] = text
#             await self.process_user_input(call_sid, text)

#     # ==================== USER INPUT ====================

#     async def process_user_input(self, call_sid, text):
#         conn = self.active_connections.get(call_sid)
#         if not conn:
#             return

#         session = conn["session"]

#         if is_question(text):
#             answer = session.answer_user_question(text)
#             await self.speak(call_sid, answer)
#             return

#         reply, advance = session.handle_response(text)
#         await self.speak(call_sid, reply)

#         if advance:
#             session.advance_step()
#             if session.get_current_step() is None:
#                 await conn["websocket"].close()

#     # ==================== SPEAK (XTTS + BARGE-IN SAFE) ====================

#     async def speak(self, call_sid, text, priority=False):
#         conn = self.active_connections.get(call_sid)
#         if not conn:
#             return

#         token = str(uuid.uuid4())
#         self.playback_tokens[call_sid] = token
#         conn["agent_speaking"] = True

#         text = trim_for_tts(text)

#         try:
#             wav = synthesize_xtts_remote(text)
#             audio = (wav * 32767).astype(np.int16).tobytes()

#             CHUNK = 320  # ~20ms
#             for i in range(0, len(audio), CHUNK):
#                 if self.playback_tokens.get(call_sid) != token:
#                     return

#                 payload = base64.b64encode(audio[i:i + CHUNK]).decode()
#                 await conn["websocket"].send(json.dumps({
#                     "event": "media",
#                     "streamSid": call_sid,
#                     "media": {"payload": payload}
#                 }))
#                 await asyncio.sleep(0.02)

#         finally:
#             conn["agent_speaking"] = False


# # ==================== SINGLETON ====================

# _voice_handler = None

# def get_voice_handler():
#     global _voice_handler
#     if not _voice_handler:
#         _voice_handler = GPUVoiceHandler()
#     return _voice_handler
