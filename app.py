"""
Production-ready Flask server for Medicaid Voice Agent
Fixed to ensure questions are asked only once per step
"""

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
import os
from dotenv import load_dotenv
from medicaid_voice_agent import CallSession
import json

load_dotenv()

app = Flask(__name__)

# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
PUBLIC_URL = os.getenv('PUBLIC_URL')

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Store active sessions
active_sessions = {}

@app.route('/make-call', methods=['POST'])
def make_call():
    """Initiate outbound call"""
    data = request.json
    to_number = data.get('to_number')
    member_id = data.get('member_id', '12345')
    
    if not to_number:
        return {'error': 'to_number is required'}, 400
    
    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f'{PUBLIC_URL}/voice',
            status_callback=f'{PUBLIC_URL}/call-status',
            status_callback_event=['completed', 'failed', 'busy', 'no-answer']
        )
        
        print(f"📞 Call initiated to {to_number} - SID: {call.sid}")
        
        return {
            'success': True,
            'call_sid': call.sid,
            'status': call.status
        }
    except Exception as e:
        print(f"❌ Error initiating call: {e}")
        return {'error': str(e)}, 500

@app.route('/voice', methods=['POST'])
def voice():
    """Initial call handler - start conversation"""
    call_sid = request.form.get('CallSid')
    
    print(f"📱 Incoming call: {call_sid}")
    
    # Create new session
    session = CallSession(call_id=call_sid)
    active_sessions[call_sid] = session
    
    # Get first question
    question = session.ask_question()
    print(f"🤖 [{call_sid}] Agent asks: {question}")
    
    return build_gather_response(call_sid, question)

@app.route('/process-speech', methods=['POST'])
def process_speech():
    """Process user's speech input"""
    call_sid = request.form.get('CallSid')
    speech_result = request.form.get('SpeechResult', '').strip()
    
    if call_sid not in active_sessions:
        print(f"❌ Session not found: {call_sid}")
        return error_response("Session not found")
    
    session = active_sessions[call_sid]
    
    print(f"👤 [{call_sid}] User: {speech_result}")
    
    # Process the response
    agent_response, should_advance = session.handle_response(speech_result)
    print(f"🤖 [{call_sid}] Agent: {agent_response}")
    
    response = VoiceResponse()
    
    # Say the agent's response
    response.say(agent_response, voice='Polly.Joanna', language='en-US')
    
    # If we should advance, move to next step
    if should_advance:
        session.advance_step()
    
    # Check current step status
    current_step = session.get_current_step()
    
    # Handle call completion
    if current_step is None:
        print(f"✅ [{call_sid}] Call complete - ending")
        save_session(call_sid, session)
        response.hangup()
        return Response(str(response), mimetype='text/xml')
    
    if current_step == "close":
        final_message = session.ask_question()
        print(f"🤖 [{call_sid}] Final message: {final_message}")
        response.say(final_message, voice='Polly.Joanna', language='en-US')
        session.form['status'] = 'REDETERMINATION_COMPLETE'
        save_session(call_sid, session)
        response.hangup()
        return Response(str(response), mimetype='text/xml')
    
    # Ask next question only if we need to
    if session.needs_question():
        next_question = session.ask_question()
        print(f"🤖 [{call_sid}] Next question: {next_question}")
        
        gather = Gather(
            input='speech',
            timeout=5,
            speech_timeout='auto',
            action=f'{PUBLIC_URL}/process-speech',
            method='POST',
            language='en-US',
            hints='yes, no, okay, sure'
        )
        gather.say(next_question, voice='Polly.Joanna', language='en-US')
        response.append(gather)
        
        # Fallback if no input
        response.redirect(url=f'{PUBLIC_URL}/retry-step?call_sid={call_sid}')
    else:
        # Just listen for more input on same question
        gather = Gather(
            input='speech',
            timeout=5,
            speech_timeout='auto',
            action=f'{PUBLIC_URL}/process-speech',
            method='POST',
            language='en-US'
        )
        response.append(gather)
        
        # Fallback
        response.redirect(url=f'{PUBLIC_URL}/retry-step?call_sid={call_sid}')
    
    return Response(str(response), mimetype='text/xml')

@app.route('/retry-step', methods=['POST', 'GET'])
def retry_step():
    """Retry current step if no input received"""
    call_sid = request.args.get('call_sid') or request.form.get('CallSid')
    
    if call_sid not in active_sessions:
        return error_response("Session not found")
    
    session = active_sessions[call_sid]
    session.retry_count += 1
    
    print(f"🔄 [{call_sid}] Retry attempt {session.retry_count}/{session.max_retries}")
    
    if session.retry_count >= session.max_retries:
        print(f"❌ [{call_sid}] Max retries reached - scheduling callback")
        response = VoiceResponse()
        response.say(
            "I'm having trouble hearing you. I'll call back later. Goodbye.",
            voice='Polly.Joanna',
            language='en-US'
        )
        session.form['status'] = 'CALLBACK_REQUIRED'
        save_session(call_sid, session)
        response.hangup()
        return Response(str(response), mimetype='text/xml')
    
    # Prompt for input without repeating full question
    response = VoiceResponse()
    
    gather = Gather(
        input='speech',
        timeout=5,
        speech_timeout='auto',
        action=f'{PUBLIC_URL}/process-speech',
        method='POST',
        language='en-US'
    )
    
    gather.say("I'm listening. Please respond.", voice='Polly.Joanna', language='en-US')
    response.append(gather)
    
    # If still no response
    response.redirect(url=f'{PUBLIC_URL}/retry-step?call_sid={call_sid}')
    
    return Response(str(response), mimetype='text/xml')

def build_gather_response(call_sid, question):
    """Helper to build a Gather TwiML response"""
    response = VoiceResponse()
    
    gather = Gather(
        input='speech',
        timeout=5,
        speech_timeout='auto',
        action=f'{PUBLIC_URL}/process-speech',
        method='POST',
        language='en-US',
        hints='yes, no, okay, sure'
    )
    
    gather.say(question, voice='Polly.Joanna', language='en-US')
    response.append(gather)
    
    # Fallback if no input
    response.redirect(url=f'{PUBLIC_URL}/retry-step?call_sid={call_sid}')
    
    return Response(str(response), mimetype='text/xml')

@app.route('/call-status', methods=['POST'])
def call_status():
    """Handle call status updates from Twilio"""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    print(f"📊 [{call_sid}] Status: {call_status}")
    
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        if call_sid in active_sessions:
            session = active_sessions[call_sid]
            
            if call_status == 'completed' and session.form['status'] == 'IN_PROGRESS':
                session.form['status'] = 'COMPLETED'
            elif call_status != 'completed':
                session.form['status'] = call_status.upper()
            
            save_session(call_sid, session)
            
            # Clean up
            del active_sessions[call_sid]
            print(f"🧹 [{call_sid}] Session cleaned up")
    
    return '', 200

def save_session(call_sid, session):
    """Save session data to file"""
    os.makedirs('call_logs', exist_ok=True)
    filepath = f'call_logs/{call_sid}.json'
    
    with open(filepath, 'w') as f:
        json.dump(session.form, f, indent=2)
    
    print(f"💾 Session saved: {filepath}")
    print(f"   Status: {session.form['status']}")
    print(f"   Member verified: {session.form['member_verified']}")
    print(f"   Form data: {json.dumps(session.form, indent=2)}")

def error_response(message):
    """Generate error TwiML response"""
    response = VoiceResponse()
    response.say(f"An error occurred: {message}", voice='Polly.Joanna', language='en-US')
    response.hangup()
    return Response(str(response), mimetype='text/xml')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'active_calls': len(active_sessions),
        'sessions': list(active_sessions.keys()),
        'config': {
            'twilio_number': TWILIO_PHONE_NUMBER,
            'public_url': PUBLIC_URL
        }
    }

@app.route('/')
def index():
    """Root endpoint"""
    return {
        'service': 'Medicaid Voice Agent',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'make_call': f'{PUBLIC_URL}/make-call',
            'health': f'{PUBLIC_URL}/health'
        }
    }

if __name__ == '__main__':
    # Create call_logs directory
    os.makedirs('call_logs', exist_ok=True)
    
    port = int(os.getenv('FLASK_PORT', 5000))
    
    print("\n" + "="*60)
    print("Medicaid Voice Agent Server Starting")
    print("="*60)
    print(f"📍 Port: {port}")
    print(f"📞 Webhook URL: {PUBLIC_URL}/voice")
    print(f"💚 Health check: http://localhost:{port}/health")
    print(f"📝 Call logs: ./call_logs/")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)