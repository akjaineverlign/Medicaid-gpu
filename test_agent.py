#!/usr/bin/env python3
"""
Interactive Agent Tester
Test the Medicaid Voice Agent without making actual phone calls
Simulates the entire conversation flow including STT, RAG, and TTS
"""

import os
import sys
import time
import json
import base64
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from colorama import init, Fore, Style
import requests

# Initialize colorama for colored terminal output
init(autoreset=True)

load_dotenv()

# ==================== TEST CONFIGURATION ====================

XTTS_SERVER_URL = os.getenv('XTTS_SERVER_URL', 'http://localhost:8000')
TEST_CALL_SID = 'TEST_CALL_123456'
TEST_MEMBER_ID = '12345'

# ==================== IMPORT COMPONENTS ====================

print(f"{Fore.CYAN}🔧 Loading components...{Style.RESET_ALL}")

try:
    from medicaid_voice_agent import CallSession, get_gpu_rag
    from gpu_rag_system import GPURAGSystem
    print(f"{Fore.GREEN}✅ Medicaid agent loaded{Style.RESET_ALL}")
except ImportError as e:
    print(f"{Fore.RED}❌ Error importing medicaid_voice_agent: {e}{Style.RESET_ALL}")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.GREEN}✅ Whisper loaded (device: {DEVICE}){Style.RESET_ALL}")
except ImportError as e:
    print(f"{Fore.RED}❌ Error importing faster_whisper: {e}{Style.RESET_ALL}")
    sys.exit(1)

# ==================== TEST UTILITIES ====================

class AgentTester:
    """Simulates a full agent conversation without phone calls"""
    
    def __init__(self):
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🧪 Initializing Agent Tester{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        # Initialize session
        print(f"{Fore.YELLOW}📋 Creating test call session...{Style.RESET_ALL}")
        self.session = CallSession(
            call_id=TEST_CALL_SID,
            member_id=TEST_MEMBER_ID
        )
        print(f"{Fore.GREEN}✅ Session created for member: {self.session.member_id}{Style.RESET_ALL}")
        
        # Initialize GPU RAG
        print(f"\n{Fore.YELLOW}📚 Loading GPU RAG system...{Style.RESET_ALL}")
        self.gpu_rag = get_gpu_rag()
        print(f"{Fore.GREEN}✅ GPU RAG ready{Style.RESET_ALL}")
        
        # Initialize Whisper (for audio testing)
        print(f"\n{Fore.YELLOW}🎤 Loading Whisper STT...{Style.RESET_ALL}")
        self.whisper = WhisperModel(
            "large-v3",
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8"
        )
        print(f"{Fore.GREEN}✅ Whisper ready{Style.RESET_ALL}")
        
        # Check XTTS server
        print(f"\n{Fore.YELLOW}🎙️  Checking XTTS server...{Style.RESET_ALL}")
        self.xtts_available = self._check_xtts()
        
        # Conversation history
        self.conversation_history = []
        
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ Agent Tester Ready!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")
    
    def _check_xtts(self):
        """Check if XTTS server is available"""
        try:
            response = requests.get(f'{XTTS_SERVER_URL}/health', timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"{Fore.GREEN}✅ XTTS server connected{Style.RESET_ALL}")
                print(f"   Device: {health.get('device')}")
                print(f"   Model: {health.get('model')}")
                return True
            else:
                print(f"{Fore.RED}❌ XTTS server returned {response.status_code}{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}❌ Cannot connect to XTTS server: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}⚠️  TTS features will be disabled{Style.RESET_ALL}")
            return False
    
    def agent_speak(self, text, test_tts=False):
        """Agent speaks (optionally test TTS)"""
        print(f"\n{Fore.BLUE}🤖 Agent: {text}{Style.RESET_ALL}")
        
        self.conversation_history.append({
            "role": "agent",
            "text": text,
            "timestamp": time.time()
        })
        
        if test_tts and self.xtts_available:
            self._test_tts(text)
    
    def _test_tts(self, text):
        """Test TTS generation"""
        try:
            print(f"{Fore.YELLOW}   🎤 Generating speech...{Style.RESET_ALL}", end='', flush=True)
            start_time = time.time()
            
            response = requests.post(
                f'{XTTS_SERVER_URL}/tts',
                json={"text": text, "language": "en"},
                params={"format": "wav"},
                timeout=30
            )
            
            if response.status_code == 200:
                elapsed = time.time() - start_time
                audio_size = len(response.content)
                print(f"\r{Fore.GREEN}   ✅ Speech generated in {elapsed:.2f}s ({audio_size:,} bytes){Style.RESET_ALL}")
                
                # Optionally save audio
                if input(f"{Fore.YELLOW}   💾 Save audio? (y/n): {Style.RESET_ALL}").lower() == 'y':
                    filename = f"test_audio_{int(time.time())}.wav"
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"{Fore.GREEN}   ✅ Saved to {filename}{Style.RESET_ALL}")
            else:
                print(f"\r{Fore.RED}   ❌ TTS failed: {response.status_code}{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"\r{Fore.RED}   ❌ TTS error: {e}{Style.RESET_ALL}")
    
    def user_respond(self, text):
        """User responds"""
        print(f"\n{Fore.GREEN}👤 You: {text}{Style.RESET_ALL}")
        
        self.conversation_history.append({
            "role": "user",
            "text": text,
            "timestamp": time.time()
        })
        
        # Check if it's a question
        from medicaid_voice_agent import is_question
        if is_question(text):
            print(f"{Fore.YELLOW}   ❓ Question detected - using GPU RAG{Style.RESET_ALL}")
            answer = self.session.answer_user_question(text)
            self.agent_speak(answer)
            return None, False
        
        # Process response
        agent_response, should_advance = self.session.handle_response(text)
        
        return agent_response, should_advance
    
    def run_conversation(self, auto_mode=False, test_tts=False):
        """Run the conversation loop"""
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🎬 Starting Conversation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        if auto_mode:
            print(f"{Fore.YELLOW}🤖 Auto mode: Using pre-defined responses{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.YELLOW}💬 Interactive mode: Type your responses{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}   Commands: 'quit' to exit, 'skip' to advance{Style.RESET_ALL}\n")
        
        # Start conversation
        first_question = self.session.ask_question()
        self.agent_speak(first_question, test_tts=test_tts)
        
        # Auto responses for testing
        auto_responses = {
            "greeting": "yes this is john",
            "identity_auth": "january 15 1985",
            "address_check": "123 main street springfield illinois 62701",
            "household_change": "no",
            "income_update": "yes",
            "other_insurance": "no"
        }
        
        income_given = False
        
        # Conversation loop
        while True:
            current_step = self.session.get_current_step()
            
            if current_step is None:
                print(f"\n{Fore.GREEN}✅ Conversation complete!{Style.RESET_ALL}")
                break
            
            if current_step == "close":
                print(f"\n{Fore.GREEN}✅ Reached closing step{Style.RESET_ALL}")
                break
            
            # Get user input
            if auto_mode:
                if self.session.awaiting_data == "new_income" and not income_given:
                    user_input = "2400"
                    income_given = True
                else:
                    user_input = auto_responses.get(current_step, "yes")
                
                print(f"\n{Fore.GREEN}👤 You (auto): {user_input}{Style.RESET_ALL}")
                time.sleep(0.5)
            else:
                user_input = input(f"\n{Fore.GREEN}👤 You: {Style.RESET_ALL}").strip()
                
                if user_input.lower() == 'quit':
                    print(f"{Fore.YELLOW}👋 Exiting conversation{Style.RESET_ALL}")
                    break
                
                if user_input.lower() == 'skip':
                    self.session.advance_step()
                    if self.session.needs_question():
                        next_q = self.session.ask_question()
                        self.agent_speak(next_q, test_tts=test_tts)
                    continue
            
            # Process response
            agent_response, should_advance = self.user_respond(user_input)
            
            if agent_response:
                self.agent_speak(agent_response, test_tts=test_tts)
            
            if should_advance:
                self.session.advance_step()
                
                next_step = self.session.get_current_step()
                
                if next_step and next_step != "close":
                    if self.session.needs_question():
                        next_q = self.session.ask_question()
                        self.agent_speak(next_q, test_tts=test_tts)
                elif next_step == "close":
                    final = self.session.ask_question()
                    self.agent_speak(final, test_tts=test_tts)
                    break
        
        # Show summary
        self._show_summary()
    
    def test_rag_question(self, question):
        """Test RAG system with a specific question"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}❓ Testing RAG Question{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        print(f"{Fore.GREEN}👤 Question: {question}{Style.RESET_ALL}")
        
        # Get conversation context
        context = self.session.get_conversation_context()
        current_step = self.session.get_current_step()
        
        print(f"\n{Fore.YELLOW}🔍 Searching knowledge base...{Style.RESET_ALL}")
        
        # Get answer
        answer = self.session.answer_user_question(question)
        
        print(f"\n{Fore.BLUE}🤖 Agent: {answer}{Style.RESET_ALL}")
        
        # Test TTS if available
        if self.xtts_available:
            if input(f"\n{Fore.YELLOW}🎤 Test TTS for this answer? (y/n): {Style.RESET_ALL}").lower() == 'y':
                self._test_tts(answer)
    
    def _show_summary(self):
        """Show conversation summary"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Conversation Summary{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        print(f"{Fore.YELLOW}Member Information:{Style.RESET_ALL}")
        print(f"  Name: {self.session.form['member_name']}")
        print(f"  Member ID: {self.session.form['member_id']}")
        print(f"  Verified: {self.session.form['member_verified']}")
        
        print(f"\n{Fore.YELLOW}Form Data:{Style.RESET_ALL}")
        print(f"  Address Confirmed: {self.session.form.get('address_confirmed', 'N/A')}")
        print(f"  Household Change: {self.session.form['household_change']['has_change']}")
        print(f"  Income Changed: {self.session.form['income_update']['changed']}")
        print(f"  Other Insurance: {self.session.form['other_insurance']['has_insurance']}")
        
        print(f"\n{Fore.YELLOW}Conversation Stats:{Style.RESET_ALL}")
        print(f"  Total exchanges: {len(self.conversation_history)}")
        print(f"  User messages: {sum(1 for m in self.conversation_history if m['role'] == 'user')}")
        print(f"  Agent messages: {sum(1 for m in self.conversation_history if m['role'] == 'agent')}")
        
        print(f"\n{Fore.YELLOW}Final Status: {self.session.form['status']}{Style.RESET_ALL}")
        
        # Save conversation log
        if input(f"\n{Fore.YELLOW}💾 Save conversation log? (y/n): {Style.RESET_ALL}").lower() == 'y':
            os.makedirs('test_logs', exist_ok=True)
            filename = f"test_logs/test_{int(time.time())}.json"
            
            log_data = {
                "session_data": self.session.form,
                "conversation_history": self.conversation_history
            }
            
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"{Fore.GREEN}✅ Saved to {filename}{Style.RESET_ALL}")


# ==================== TEST SCENARIOS ====================

def test_full_conversation():
    """Test complete conversation flow"""
    tester = AgentTester()
    
    mode = input(f"\n{Fore.YELLOW}Choose mode (1=Interactive, 2=Auto): {Style.RESET_ALL}").strip()
    auto_mode = mode == '2'
    
    test_tts = input(f"{Fore.YELLOW}Test TTS generation? (y/n): {Style.RESET_ALL}").lower() == 'y'
    
    tester.run_conversation(auto_mode=auto_mode, test_tts=test_tts)


def test_rag_only():
    """Test RAG system only"""
    tester = AgentTester()
    
    print(f"\n{Fore.CYAN}🧪 RAG Testing Mode{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type questions to test the RAG system{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'quit' to exit{Style.RESET_ALL}\n")
    
    common_questions = [
        "Why do you need my address?",
        "What happens if I don't complete this?",
        "Can I call back later?",
        "What if my income changed?",
        "Will I lose my coverage?",
    ]
    
    print(f"{Fore.CYAN}Example questions:{Style.RESET_ALL}")
    for i, q in enumerate(common_questions, 1):
        print(f"  {i}. {q}")
    print()
    
    while True:
        question = input(f"{Fore.GREEN}Your question (or number 1-5, or 'quit'): {Style.RESET_ALL}").strip()
        
        if question.lower() == 'quit':
            break
        
        if question.isdigit() and 1 <= int(question) <= len(common_questions):
            question = common_questions[int(question) - 1]
        
        if question:
            tester.test_rag_question(question)


def test_components():
    """Test individual components"""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🔧 Component Testing{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # Test GPU RAG
    print(f"{Fore.YELLOW}1. Testing GPU RAG System...{Style.RESET_ALL}")
    try:
        rag = GPURAGSystem()
        results = rag.search("why do you need my address", k=3)
        print(f"{Fore.GREEN}✅ GPU RAG working - found {len(results)} results{Style.RESET_ALL}")
        if results:
            print(f"   Top result confidence: {results[0]['confidence']:.2f}")
    except Exception as e:
        print(f"{Fore.RED}❌ GPU RAG error: {e}{Style.RESET_ALL}")
    
    # Test Whisper
    print(f"\n{Fore.YELLOW}2. Testing Whisper STT...{Style.RESET_ALL}")
    try:
        whisper = WhisperModel("large-v3", device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")
        
        # Create test audio (1 second of sine wave)
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        segments, info = whisper.transcribe(test_audio, language="en")
        
        print(f"{Fore.GREEN}✅ Whisper working{Style.RESET_ALL}")
        print(f"   Device: {DEVICE}")
        print(f"   Language detected: {info.language}")
    except Exception as e:
        print(f"{Fore.RED}❌ Whisper error: {e}{Style.RESET_ALL}")
    
    # Test XTTS
    print(f"\n{Fore.YELLOW}3. Testing XTTS Server...{Style.RESET_ALL}")
    try:
        response = requests.get(f'{XTTS_SERVER_URL}/health', timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"{Fore.GREEN}✅ XTTS server working{Style.RESET_ALL}")
            print(f"   URL: {XTTS_SERVER_URL}")
            print(f"   Device: {health.get('device')}")
            
            # Test generation
            print(f"\n{Fore.YELLOW}   Testing speech generation...{Style.RESET_ALL}")
            start = time.time()
            resp = requests.post(
                f'{XTTS_SERVER_URL}/tts',
                json={"text": "This is a test.", "language": "en"},
                params={"format": "wav"},
                timeout=30
            )
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                print(f"{Fore.GREEN}   ✅ Speech generated in {elapsed:.2f}s ({len(resp.content):,} bytes){Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}   ❌ Generation failed: {resp.status_code}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ XTTS server error: {response.status_code}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ XTTS server error: {e}{Style.RESET_ALL}")
    
    # Test Call Session
    print(f"\n{Fore.YELLOW}4. Testing Call Session...{Style.RESET_ALL}")
    try:
        session = CallSession(call_id="TEST_123", member_id="12345")
        question = session.ask_question()
        print(f"{Fore.GREEN}✅ Call session working{Style.RESET_ALL}")
        print(f"   First question: {question[:50]}...")
    except Exception as e:
        print(f"{Fore.RED}❌ Call session error: {e}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✅ Component testing complete{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")


# ==================== MAIN MENU ====================

def main():
    """Main test menu"""
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧪 Medicaid Voice Agent - Test Suite{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    print(f"{Fore.YELLOW}Choose a test:{Style.RESET_ALL}")
    print(f"  1. Test full conversation (interactive or auto)")
    print(f"  2. Test RAG system only")
    print(f"  3. Test individual components")
    print(f"  4. Quick component check")
    print(f"  5. Exit")
    
    choice = input(f"\n{Fore.GREEN}Your choice (1-5): {Style.RESET_ALL}").strip()
    
    if choice == '1':
        test_full_conversation()
    elif choice == '2':
        test_rag_only()
    elif choice == '3':
        test_components()
    elif choice == '4':
        # Quick check
        print(f"\n{Fore.YELLOW}Running quick check...{Style.RESET_ALL}\n")
        tester = AgentTester()
        print(f"\n{Fore.GREEN}✅ All components loaded successfully!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}First question from agent:{Style.RESET_ALL}")
        q = tester.session.ask_question()
        print(f"{Fore.BLUE}{q}{Style.RESET_ALL}")
    elif choice == '5':
        print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}👋 Test interrupted{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()