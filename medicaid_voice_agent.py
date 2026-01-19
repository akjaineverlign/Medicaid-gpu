"""
Updated Medicaid Voice Agent with GPU RAG Integration
Now properly connected to gpu_rag_system.py
"""

import json
import re
from openai import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
import random

load_dotenv()

from gpu_rag_system import GPURAGSystem, answer_with_rag_and_context

# Initialize GPU RAG system (singleton)
print("🔧 Initializing GPU RAG System...")
gpu_rag = GPURAGSystem()
print("✅ GPU RAG System initialized")

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
MODEL = os.getenv("LLM_MODEL")

def llm(prompt, json_mode=False):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_completion_tokens": 250,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    res = client.chat.completions.create(**payload)
    return res.choices[0].message.content.strip()

MEMBER_DB = {
    "12345": {
        "name": "John Doe",
        "dob": "01/15/1985",
        "address": "123 Main Street, Springfield, IL 62701",
        "household_income": 2100.00,
        "household_size": 2
    }
}

def fuzzy_match_address(user_address, db_address, threshold=70):
    """Match address with fuzzy logic"""
    user_clean = user_address.lower().strip()
    db_clean = db_address.lower().strip()
    
    similarity = fuzz.ratio(user_clean, db_clean)
    
    # Parse address components
    db_parts = {
        'street': '',
        'city': '',
        'state': '',
        'zip': ''
    }
    
    db_components = db_address.split(',')
    if len(db_components) >= 1:
        db_parts['street'] = db_components[0].strip()
    if len(db_components) >= 2:
        db_parts['city'] = db_components[1].strip()
    if len(db_components) >= 3:
        state_zip = db_components[2].strip().split()
        if len(state_zip) >= 1:
            db_parts['state'] = state_zip[0].strip()
        if len(state_zip) >= 2:
            db_parts['zip'] = state_zip[1].strip()
    
    # Check what's missing
    missing = []
    if db_parts['street'] and db_parts['street'].lower() not in user_clean:
        street_number = re.search(r'\d+', db_parts['street'])
        if street_number and street_number.group() not in user_address:
            missing.append(f"street number (should start with {db_parts['street'].split()[0]})")
    
    if db_parts['city'] and db_parts['city'].lower() not in user_clean:
        missing.append(f"city ({db_parts['city']})")
    
    if db_parts['state'] and db_parts['state'].lower() not in user_clean:
        missing.append(f"state ({db_parts['state']})")
    
    if db_parts['zip'] and db_parts['zip'] not in user_address:
        missing.append(f"zip code ({db_parts['zip']})")
    
    return similarity >= threshold, similarity, missing, db_parts

# ==================== CALL SESSION CLASS ====================
class CallSession:
    """Manages state for a single call - Now integrated with GPU RAG"""
    
    SCRIPT = [
        "greeting",
        "identity_auth",
        "address_check",
        "household_change",
        "income_update",
        "other_insurance",
        "close"
    ]
    
    def __init__(self, call_id, member_id="12345"):
        self.call_id = call_id
        self.member_id = member_id
        self.current_step_index = 0
        self.awaiting_data = None
        self.retry_count = 0
        self.max_retries = 3
        self.question_asked = False
        self.address_attempts = 0
        self.user_provided_address = None
        
        # Conversation context for RAG
        self.conversation_history = []
        
        member = MEMBER_DB.get(member_id, {})
        
        self.form = {
            "call_id": call_id,
            "member_id": member_id,
            "member_name": member.get("name", "Unknown"),
            "member_verified": False,
            "verified_dob": None,
            
            "address_confirmed": None,
            "current_address": member.get("address"),
            "user_provided_address": None,
            "address_corrections_made": [],
            "new_address": None,
            
            "household_change": {
                "has_change": False,
                "details": None
            },
            
            "income_update": {
                "changed": False,
                "previous_amount": member.get("household_income"),
                "new_amount": None,
                "frequency": "monthly"
            },
            
            "other_insurance": {
                "has_insurance": False,
                "details": None
            },
            
            "status": "IN_PROGRESS",
            "timestamp": datetime.now().isoformat()
        }
    
    def add_to_conversation(self, role, message):
        """Add message to conversation history for RAG context"""
        self.conversation_history.append({
            "role": role,
            "content": message
        })
    
    def get_conversation_context(self):
        """Format conversation for RAG"""
        formatted = []
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Agent"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def get_member(self):
        return MEMBER_DB.get(self.member_id)
    
    def get_current_step(self):
        if self.current_step_index >= len(self.SCRIPT):
            return None
        return self.SCRIPT[self.current_step_index]
    
    def advance_step(self):
        self.current_step_index += 1
        self.awaiting_data = None
        self.retry_count = 0
        self.question_asked = False
        self.address_attempts = 0
        self.user_provided_address = None
    
    def ask_question(self):
        """Get question for current step"""
        step = self.get_current_step()
        member = self.get_member()
        
        self.question_asked = True
        
        questions = {
            "greeting": f"Hi there! This is the Medicaid Renewal Assistant calling for {member['name']}. Is this {member['name']}?",
            "identity_auth": "Great! Before we get started, I need to verify your identity for security. Could you please tell me your date of birth?",
            "address_check": "Perfect. Now, could you please tell me your current mailing address? I need the street, city, state, and zip code.",
            "household_change": "Alright, next question. Has anyone moved in or out of your household in the past year? For example, a spouse, child, or anyone else?",
            "income_update": f"Got it. Our records show your household income is about ${member['household_income']:.2f} per month. Has that changed at all?",
            "other_insurance": "Last question! Do you or anyone in your household have health insurance through work or another private plan?",
            "close": "Perfect! We're all done. I've updated your information, and you'll get a confirmation letter in the mail within 10 days. Thanks so much for your time, and have a wonderful day!"
        }
        
        question = questions.get(step, "")
        
        # Add to conversation history
        if question:
            self.add_to_conversation("agent", question)
        
        return question
    
    def needs_question(self):
        """Check if current step needs a question asked"""
        return not self.question_asked
    
    def answer_user_question(self, user_question):
        """
        Use GPU RAG to answer user's question with full conversation context
        This is the key integration point!
        """
        current_step = self.get_current_step()
        conversation_context = self.get_conversation_context()
        
        print(f"🔍 Using GPU RAG to answer: {user_question}")
        print(f"📝 With conversation context ({len(self.conversation_history)} messages)")
        
        # Use GPU RAG system to answer
        answer = answer_with_rag_and_context(
            gpu_rag=gpu_rag,
            question=user_question,
            conversation_history=conversation_context,
            current_step=current_step
        )
        
        print(f"✅ GPU RAG answer: {answer}")
        
        # Add to conversation history
        self.add_to_conversation("user", user_question)
        self.add_to_conversation("agent", answer)
        
        return answer
    
    def handle_response(self, user_input):
        """
        Process user response
        Now checks for questions and uses GPU RAG!
        """
        step = self.get_current_step()
        member = self.get_member()
        
        if not user_input or user_input.strip() == "":
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.form["status"] = "CALLBACK_REQUIRED"
                return "I'm having trouble hearing you. Let me call back at a better time. Take care!", True
            return "Sorry, I didn't catch that. Could you say that again?", False
        
        self.retry_count = 0
        
        # Add user input to conversation history
        self.add_to_conversation("user", user_input)
        
        # This is handled by voice handler now, but keeping for compatibility
        
        # Handle specific steps
        if step == "greeting":
            response, should_advance = self._handle_greeting(user_input, member)
        elif step == "identity_auth":
            response, should_advance = self._handle_identity(user_input, member)
        elif step == "address_check":
            response, should_advance = self._handle_address(user_input, member)
        elif step == "household_change":
            response, should_advance = self._handle_household(user_input)
        elif step == "income_update":
            response, should_advance = self._handle_income(user_input, member)
        elif step == "other_insurance":
            response, should_advance = self._handle_insurance(user_input)
        else:
            response = "I didn't quite get that. Could you please say that again?"
            should_advance = False
        
        # Add response to conversation history
        if response:
            self.add_to_conversation("agent", response)
        
        return response, should_advance
    
    def _handle_greeting(self, user_input, member):
        if is_affirmative(user_input):
            self.form["member_verified"] = True
            return "Perfect! Let's get started with your renewal.", True
        elif is_negative(user_input):
            self.form["member_verified"] = False
            return f"Oh, I apologize! May I speak with {member['name']}, or should I call back later?", False
        else:
            return f"Sorry, I didn't catch that. Am I speaking with {member['name']}?", False
    
    def _handle_identity(self, user_input, member):
        parsed_dob = normalize_date(user_input)
        
        if parsed_dob and validate_dob(parsed_dob, member['dob']):
            self.form["verified_dob"] = parsed_dob
            return "Perfect, that matches our records.", True
        elif parsed_dob:
            return "Hmm, that doesn't match what I have on file. Could you please verify your date of birth again?", False
        else:
            return "I didn't quite catch the date. Could you say your date of birth? For example, 'January 15, 1985'.", False
    
    def _handle_address(self, user_input, member):
        """Address verification with proper multi-turn handling"""

        db_address = member["address"]


        if self.awaiting_data == "confirm_corrected_address":
            if is_affirmative(user_input):
                self.form["address_confirmed"] = "confirmed"
                self.awaiting_data = None
                return "Perfect, I've confirmed your address.", True

            if is_negative(user_input):
                self.awaiting_data = "new_address"
                return (
                    "No problem. What's your full current address including street, city, state, and zip?",
                    False
                )

            return "Just to confirm, is that address correct? Please say yes or no.", False

        if self.awaiting_data == "new_address":
            new_address = extract_address(user_input)

            if len(new_address) > 15:
                self.form["address_confirmed"] = "changed"
                self.form["new_address"] = new_address
                self.awaiting_data = None
                return "Got it, I've updated your address.", True

            return "Could you please give the full address with street, city, state, and zip?", False


        user_address = extract_address(user_input)
        self.address_attempts += 1

        self.user_provided_address = user_address
        self.form["user_provided_address"] = user_address

        is_match, similarity, missing_parts, _ = fuzzy_match_address(
            user_address, db_address
        )

        print(f"[ADDRESS] similarity={similarity}, missing={missing_parts}")

        if similarity >= 90:
            self.form["address_confirmed"] = "confirmed"
            return "Great, that matches perfectly.", True

        if similarity >= 70 and missing_parts:
            self.awaiting_data = "confirm_corrected_address"

            self.form["address_corrections_made"].append({
                "user_provided": user_address,
                "corrected_to": db_address,
                "similarity": similarity
            })

            return f"I have your address as {db_address}. Is that correct?", False

        if similarity >= 50:
            return f"I have your address on file as {db_address}. Is that still correct?", False


        if self.address_attempts >= 2:
            self.form["address_confirmed"] = "changed"
            self.form["new_address"] = user_address
            return "Thanks, I've updated your address in the system.", True

        return (
            f"That doesn't match our records. We have {db_address}. Has your address changed?",
            False
        )

    
    def _handle_household(self, user_input):
        if self.awaiting_data == "household_details":
            self.form["household_change"]["has_change"] = True
            self.form["household_change"]["details"] = user_input
            self.awaiting_data = None
            return "Got it, thank you. I've noted that change.", True
        
        if is_negative(user_input):
            self.form["household_change"]["has_change"] = False
            self.form["household_change"]["details"] = "No changes"
            return "Okay, no changes.", True
        elif is_affirmative(user_input):
            self.awaiting_data = "household_details"
            return "Okay, could you tell me who moved in or out and their relationship to you?", False
        else:
            return "Has anyone moved in or out? Just yes or no.", False
    
    def _handle_income(self, user_input, member):
        if self.awaiting_data == "new_income":
            new_income = normalize_income(user_input)
            if new_income is not None:
                valid, result = validate_income(new_income)
                if valid:
                    self.form["income_update"]["changed"] = True
                    self.form["income_update"]["new_amount"] = result
                    self.awaiting_data = None
                    return "Perfect, I've recorded your new income.", True
                else:
                    return result, False
            else:
                return "I didn't catch the amount. What's your monthly income as a number, like 2400?", False
        
        if is_negative(user_input):
            self.form["income_update"]["changed"] = False
            self.form["income_update"]["new_amount"] = member['household_income']
            return "Great, I'll keep that the same.", True
        elif is_affirmative(user_input):
            self.awaiting_data = "new_income"
            return "Okay. What's your new monthly income before taxes?", False
        else:
            return "Has your income changed? Just yes or no.", False
    
    def _handle_insurance(self, user_input):
        if is_affirmative(user_input):
            self.form["other_insurance"]["has_insurance"] = True
            return "Got it. I've noted that you have other insurance.", True
        elif is_negative(user_input):
            self.form["other_insurance"]["has_insurance"] = False
            return "Okay, understood.", True
        else:
            return "Do you have other health insurance? Yes or no.", False


# ==================== HELPER FUNCTIONS ====================

def is_question(text):
    """Detect if text is a question"""
    question_words = [
        "what", "who", "why", "how", "when", "where", 
        "is", "can", "could", "would", "should", "do", "does", "will"
    ]
    text_lower = text.lower().strip()
    return text.endswith("?") or any(text_lower.startswith(w) for w in question_words)

def is_affirmative(text):
    """Detect affirmative responses"""
    affirmatives = [
        "yes", "y", "yeah", "yep", "yup", "sure", "correct", 
        "right", "that's right", "ok", "okay", "mhm", "uh huh", "definitely"
    ]
    return any(aff in text.lower().strip() for aff in affirmatives)

def is_negative(text):
    """Detect negative responses"""
    negatives = [
        "no", "n", "nope", "nah", "not", "incorrect", 
        "wrong", "uh uh", "negative"
    ]
    return any(neg in text.lower().strip() for neg in negatives)

def normalize_date(text):
    """Parse various date formats"""
    text = text.strip().lower()
    text = text.replace("the", "").replace("of", "").strip()
    
    months = {
        'january': '01', 'jan': '01',
        'february': '02', 'feb': '02',
        'march': '03', 'mar': '03',
        'april': '04', 'apr': '04',
        'may': '05',
        'june': '06', 'jun': '06',
        'july': '07', 'jul': '07',
        'august': '08', 'aug': '08',
        'september': '09', 'sep': '09', 'sept': '09',
        'october': '10', 'oct': '10',
        'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    numeric_patterns = [
        r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})',
        r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})',
    ]
    
    for pattern in numeric_patterns:
        match = re.search(pattern, text)
        if match:
            month, day, year = match.groups()
            if len(year) == 2:
                year = '19' + year if int(year) > 30 else '20' + year
            return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
    
    for month_name, month_num in months.items():
        pattern = rf'{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})'
        match = re.search(pattern, text)
        if match:
            day = match.group(1).zfill(2)
            year = match.group(2)
            return f"{month_num}/{day}/{year}"
        
        pattern = rf'(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_name},?\s+(\d{{4}})'
        match = re.search(pattern, text)
        if match:
            day = match.group(1).zfill(2)
            year = match.group(2)
            return f"{month_num}/{day}/{year}"
    
    return None

def normalize_income(text):
    """Parse income from text"""
    text = text.lower().replace(",", "").replace("$", "").strip()
    
    if any(w in text for w in ["zero", "none", "nothing", "no income"]):
        return 0.0
    
    if "lakh" in text:
        n = re.findall(r"\d+(?:\.\d+)?", text)
        return float(n[0]) * 100000 if n else None
    
    if "k" in text:
        n = re.findall(r"\d+(?:\.\d+)?", text)
        return float(n[0]) * 1000 if n else None
    
    n = re.findall(r"\d+(?:\.\d+)?", text)
    return float(n[0]) if n else None

def extract_address(text):
    """Extract address from text"""
    return text.strip()

def validate_income(val):
    """Validate income value"""
    if val < 0:
        return False, "Income can't be negative. What's the correct amount?"
    if val > 100_000:
        return False, "That seems really high. Could you verify that?"
    return True, val

def validate_dob(input_dob, stored_dob):
    """Compare date of birth"""
    input_clean = re.sub(r'[^\d]', '', input_dob)
    stored_clean = re.sub(r'[^\d]', '', stored_dob)
    return input_clean == stored_clean


# ==================== EXPORT FOR USE ====================

def get_gpu_rag():
    """Get the initialized GPU RAG system"""
    return gpu_rag