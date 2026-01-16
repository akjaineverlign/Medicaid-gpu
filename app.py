"""
Integrated Medicaid Voice Agent Server
Connects to external XTTS server via HTTP
All components communicate through network requests
"""

from flask import Flask, request, Response, jsonify
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
import os
from dotenv import load_dotenv
from medicaid_voice_agent import CallSession
import json
import asyncio
from quart import Quart, websocket
from hypercorn.config import Config
from hypercorn.asyncio import serve
import threading

load_dotenv()

# ==================== CONFIGURATION ====================

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
PUBLIC_URL = os.getenv('PUBLIC_URL')  # Your ngrok URL for webhooks

# XTTS Server Configuration
XTTS_SERVER_URL = os.getenv('XTTS_SERVER_URL', 'http://localhost:8000')

HTTP_PORT = 5000  # Flask HTTP server
WS_PORT = 5001    # WebSocket server

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Store active sessions
active_sessions = {}

# ==================== FLASK HTTP SERVER ====================

app = Flask(__name__)

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
    
    # Connect to WebSocket stream
    connect = Connect()
    connect.stream(
        url=f'wss://{PUBLIC_URL.replace("https://", "").replace("http://", "")}:{WS_PORT}/stream',
        track='inbound_track'
    )
    response.append(connect)
    
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
    # Check XTTS server connectivity
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
        'http_port': HTTP_PORT,
        'ws_port': WS_PORT,
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
        'version': '2.0',
        'status': 'running',
        'endpoints': {
            'make_call': f'{PUBLIC_URL}/make-call',
            'health': f'{PUBLIC_URL}/health',
            'websocket': f'wss://{PUBLIC_URL.replace("https://", "")}:{WS_PORT}/stream'
        },
        'xtts_server': XTTS_SERVER_URL
    }


# ==================== QUART WEBSOCKET SERVER ====================

quart_app = Quart(__name__)

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


@quart_app.websocket('/stream')
async def stream():
    """WebSocket endpoint for Twilio Media Streams"""
    print("🔌 WebSocket connection established")
    
    call_sid = None
    session = None
    handler = get_voice_handler()
    
    try:
        async for message in websocket.receive():
            data = json.loads(message)
            
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
                
                # Start handling the call
                asyncio.create_task(
                    handler.handle_call(websocket._get_current_object(), call_sid, session)
                )
            
            elif event == 'media':
                # Audio data from user
                # Handled by voice handler
                pass
            
            elif event == 'stop':
                # Call ended
                print(f"🛑 Stream stopped for call: {call_sid}")
                if call_sid and call_sid in active_sessions:
                    save_session(call_sid, active_sessions[call_sid])
                break
    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    
    finally:
        print(f"🔌 WebSocket connection closed: {call_sid}")


@quart_app.route('/ws-health')
async def ws_health():
    """WebSocket server health check"""
    return {
        'status': 'healthy',
        'service': 'websocket',
        'port': WS_PORT
    }


# ==================== SERVER STARTUP ====================

def run_flask():
    """Run Flask HTTP server"""
    print(f"\n🌐 Starting Flask HTTP server on port {HTTP_PORT}...")
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False, use_reloader=False)


async def run_quart():
    """Run Quart WebSocket server"""
    print(f"\n🔌 Starting WebSocket server on port {WS_PORT}...")
    config = Config()
    config.bind = [f"0.0.0.0:{WS_PORT}"]
    await serve(quart_app, config)


def main():
    """Start all servers"""
    
    print("\n" + "="*60)
    print("🚀 Medicaid Voice Agent - Integrated Server")
    print("="*60)
    print(f"📞 Twilio Number: {TWILIO_PHONE_NUMBER}")
    print(f"🌐 Public URL: {PUBLIC_URL}")
    print(f"🖥️  HTTP Server: http://localhost:{HTTP_PORT}")
    print(f"🔌 WebSocket Server: ws://localhost:{WS_PORT}")
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
        print(f"   Start it with: python xtts_server.py")
    
    print("\n✅ All components initialized")
    print("\n" + "="*60)
    print("🎯 Ready to receive calls!")
    print("="*60 + "\n")
    
    # Start Flask in separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Run WebSocket server in main thread
    asyncio.run(run_quart())


if __name__ == '__main__':
    os.makedirs('call_logs', exist_ok=True)
    main()