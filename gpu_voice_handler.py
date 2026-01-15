"""
GPU Voice Handler with Local XTTS Integration
Everything runs on same machine - no external dependencies
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
import soundfile as sf
from io import BytesIO

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

# ==================== LOCAL XTTS CLIENT ====================

class LocalXTTSClient:
    """
    Local XTTS client - runs in same process
    No network calls needed!
    """
    
    def __init__(self, voice_sample_path=None):
        print("🔧 Initializing Local XTTS Client...")
        try:
            from TTS.api import TTS
            
            self.device = DEVICE
            print(f"📱 Loading XTTS model on {self.device}...")
            
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            
            self.voice_sample_path = voice_sample_path
            if self.voice_sample_path and os.path.exists(self.voice_sample_path):
                print(f"🎤 Voice sample loaded: {self.voice_sample_path}")
            else:
                print("⚠️ No voice sample - using default voice")
            
            print("✅ Local XTTS ready")
            
        except Exception as e:
            print(f"❌ Error loading XTTS: {e}")
            self.tts = None
    
    def generate_speech(self, text, language="en"):
        """
        Generate speech locally
        Returns: Audio as numpy array
        """
        if self.tts is None:
            print("❌ XTTS not available")
            return None
        
        try:
            print(f"🎤 Generating speech: '{text[:50]}...'")
            start_time = time.time()
            
            if self.voice_sample_path and os.path.exists(self.voice_sample_path):
                wav = self.tts.tts(
                    text=text,
                    speaker_wav=self.voice_sample_path,
                    language=language
                )
            else:
                wav = self.tts.tts(
                    text=text,
                    language=language
                )
            
            elapsed = time.time() - start_time
            print(f"✅ Speech generated in {elapsed:.2f}s")
            
            return np.array(wav, dtype=np.float32)
        
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None

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
    Enhanced voice handler with local XTTS
    All processing on same machine
    """

    def __init__(self, voice_sample_path=None):
        print("🔧 Initializing GPU Voice Handler...")
        
        # Initialize Faster Whisper (GPU-accelerated STT)
        print("🔥 Loading Faster Whisper model...")
        self.whisper = WhisperModel(
            "large-v3",
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8",
            vad_filter=True
        )
        print("✅ Whisper loaded with VAD")
        
        # Initialize Local XTTS
        print("🎤 Initializing local XTTS...")
        self.tts = LocalXTTSClient(voice_sample_path=voice_sample_path)
        print("✅ XTTS initialized")
        
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
                print(f"💤 [{call_sid}] User: {transcript}")
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
        print(f"💤 [{call_sid}] User: {transcript}")

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
        """Generate and stream speech using local XTTS"""
        
        text = trim_for_tts(text)
        
        conn = self.active_connections.get(call_sid)
        if not conn:
            return

        try:
            if not low_volume:
                conn["agent_speaking"] = True

            # Generate speech locally
            audio_data = self.tts.generate_speech(text=text, language="en")
            
            if audio_data is None:
                print(f"❌ Failed to generate speech")
                conn["agent_speaking"] = False
                return

            # Convert to mu-law 8kHz for Twilio
            audio_8k = self.resample_to_8khz(audio_data)
            mulaw = self.convert_to_mulaw(audio_8k)

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

    def resample_to_8khz(self, audio, orig_sr=22050):
        """Resample audio to 8kHz"""
        import scipy.signal as signal
        
        target_sr = 8000
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)

    def convert_to_mulaw(self, audio):
        """Convert float32 audio to mu-law"""
        # Normalize
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        pcm = (audio * 32767).astype(np.int16)
        
        # Convert to mu-law
        import audioop
        mulaw = audioop.lin2ulaw(pcm.tobytes(), 2)
        
        return mulaw

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

def get_voice_handler():
    """Get or create voice handler"""
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
