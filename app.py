"""
Unified Medicaid Voice Agent Server
Single Flask-SocketIO server - no separate WebSocket server needed
"""

from flask import Flask, request, Response, jsonify
from flask_socketio import SocketIO, emit
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
import os
from dotenv import load_dotenv
from medicaid_voice_agent import CallSession
import json
import base64
import asyncio
from threading import Thread
import time

load_dotenv()

# ==================== CONFIGURATION ====================

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
PUBLIC_URL = os.getenv('PUBLIC_URL')  # Your ngrok URL

# XTTS Server Configuration
XTTS_SERVER_URL = os.getenv('XTTS_SERVER_URL', 'http://localhost:8000')

PORT = int(os.getenv('FLASK_PORT', 5000))

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Store active sessions
active_sessions = {}

# ==================== FLASK + SOCKETIO SETUP ====================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medicaid-voice-agent-secret'

# Initialize SocketIO with proper CORS settings
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True
)

# Import voice handler (will be initialized on first use)
voice_handler = None

def get_voice_handler():
    """Lazy initialization of voice handler"""
    global voice_handler
    if voice_handler is None:
        from gpu_voice_handler import GPUVoiceHandler
        print("🔧 Initializing GPU Voice Handler...")
        voice_handler = GPUVoiceHandler(xtts_server_url=XTTS_SERVER_URL)
        print("✅ Voice Handler ready")
    return voice_handler


# ==================== HTTP ENDPOINTS ====================

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
    """Initial call handler - Connect to WebSocket"""
    call_sid = request.form.get('CallSid')
    
    print(f"📱 Incoming call: {call_sid}")
    
    # Create new session
    session = CallSession(call_id=call_sid)
    active_sessions[call_sid] = session
    
    # Create TwiML response that connects to WebSocket
    response = VoiceResponse()
    
    # Initial greeting
    response.say(
        "Connecting you now, please hold.",
        voice='Polly.Joanna',
        language='en-US'
    )
    
    # Connect to WebSocket stream - SAME SERVER, SAME PORT
    connect = Connect()
    
    # Use WSS (secure WebSocket) if PUBLIC_URL uses HTTPS
    ws_protocol = 'wss' if 'https' in PUBLIC_URL else 'ws'
    ws_url = PUBLIC_URL.replace('https://', '').replace('http://', '')
    
    connect.stream(
        url=f'{ws_protocol}://{ws_url}/stream',
        track='inbound_track'
    )
    response.append(connect)
    
    print(f"📡 WebSocket URL: {ws_protocol}://{ws_url}/stream")
    
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


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    import requests
    try:
        xtts_response = requests.get(f'{XTTS_SERVER_URL}/health', timeout=2)
        xtts_healthy = xtts_response.status_code == 200
        xtts_info = xtts_response.json() if xtts_healthy else {}
    except:
        xtts_healthy = False
        xtts_info = {}
    
    return {
        'status': 'healthy',
        'active_calls': len(active_sessions),
        'sessions': list(active_sessions.keys()),
        'port': PORT,
        'xtts_server': {
            'url': XTTS_SERVER_URL,
            'healthy': xtts_healthy,
            'info': xtts_info
        },
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
        'version': '3.0',
        'status': 'running',
        'endpoints': {
            'make_call': f'{PUBLIC_URL}/make-call',
            'health': f'{PUBLIC_URL}/health',
            'websocket': f'{PUBLIC_URL}/stream'
        },
        'xtts_server': XTTS_SERVER_URL
    }


# ==================== WEBSOCKET HANDLERS ====================

@socketio.on('connect', namespace='/stream')
def handle_connect():
    """WebSocket connection established"""
    print(f"🔌 WebSocket connected: {request.sid}")


@socketio.on('disconnect', namespace='/stream')
def handle_disconnect():
    """WebSocket disconnected"""
    print(f"🔌 WebSocket disconnected: {request.sid}")


@socketio.on('message', namespace='/stream')
def handle_message(message):
    """Handle Twilio Media Stream messages"""
    try:
        data = json.loads(message) if isinstance(message, str) else message
        
        event = data.get('event')
        
        if event == 'start':
            # Call started
            call_sid = data['start']['callSid']
            print(f"🎬 Stream started for call: {call_sid}")
            
            # Get or create session
            if call_sid in active_sessions:
                session = active_sessions[call_sid]
            else:
                session = CallSession(call_id=call_sid)
                active_sessions[call_sid] = session
            
            # Start handling the call in a separate thread
            handler = get_voice_handler()
            
            def run_handler():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    handler.handle_call_socketio(request.sid, call_sid, session, socketio)
                )
            
            thread = Thread(target=run_handler, daemon=True)
            thread.start()
        
        elif event == 'media':
            # Audio data - will be processed by handler
            pass
        
        elif event == 'stop':
            # Call ended
            call_sid = data.get('callSid', 'unknown')
            print(f"🛑 Stream stopped for call: {call_sid}")
            if call_sid in active_sessions:
                save_session(call_sid, active_sessions[call_sid])
    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()


# ==================== SERVER STARTUP ====================

def main():
    """Start the server"""
    
    print("\n" + "="*60)
    print("🚀 Medicaid Voice Agent - Unified Server")
    print("="*60)
    print(f"📞 Twilio Number: {TWILIO_PHONE_NUMBER}")
    print(f"🌐 Public URL: {PUBLIC_URL}")
    print(f"🖥️  Server Port: {PORT}")
    print(f"🎤 XTTS Server: {XTTS_SERVER_URL}")
    print("="*60)
    
    # Pre-initialize components
    print("\n🔧 Initializing components...")
    
    # Initialize GPU RAG
    print("📚 Loading GPU RAG system...")
    from gpu_rag_system import GPURAGSystem
    rag = GPURAGSystem()
    print("✅ GPU RAG ready")
    
    # Check XTTS connectivity
    print(f"🎤 Checking XTTS server at {XTTS_SERVER_URL}...")
    import requests
    try:
        response = requests.get(f'{XTTS_SERVER_URL}/health', timeout=5)
        if response.status_code == 200:
            print("✅ XTTS server connected")
            print(f"   {response.json()}")
        else:
            print("⚠️  XTTS server not responding properly")
    except Exception as e:
        print(f"❌ Cannot connect to XTTS server: {e}")
        print(f"   Make sure XTTS server is running at {XTTS_SERVER_URL}")
        print(f"   Start it with: python server.py")
    
    print("\n✅ All components initialized")
    print("\n" + "="*60)
    print("🎯 Ready to receive calls!")
    print("="*60 + "\n")
    print("💡 Make sure to expose this server with ngrok:")
    print(f"   ngrok http {PORT}")
    print(f"\n   Then update PUBLIC_URL in .env to your ngrok URL")
    print("")
    
    # Start the server
    socketio.run(
        app,
        host='0.0.0.0',
        port=PORT,
        debug=False,
        allow_unsafe_werkzeug=True
    )


if __name__ == '__main__':
    os.makedirs('call_logs', exist_ok=True)
    main()