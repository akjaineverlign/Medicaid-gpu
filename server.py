"""
XTTS TTS Server for Mac
Runs locally on Mac and exposes API via ngrok
Supports voice cloning with speaker samples
"""
 
from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import torch
import io
import numpy as np
import soundfile as sf
import os
from werkzeug.utils import secure_filename

from TTS.utils.manage import ModelManager
manager = ModelManager()
print(manager.list_models())
# manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

 
app = Flask(__name__)
 
# Configuration
UPLOAD_FOLDER = 'voice_samples'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}
DEFAULT_VOICE_SAMPLE = os.getenv('DEFAULT_VOICE_SAMPLE', 'voice_sample.wav')
 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
# Initialize XTTS
print("🔧 Loading XTTS model...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"📱 Using device: {device}")
 
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("✅ XTTS model loaded")
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
def convert_to_mulaw_8khz(audio_data, sample_rate):
    """
    Convert audio to mu-law 8kHz for Twilio
    Args:
        audio_data: numpy array of audio samples
        sample_rate: original sample rate
    Returns:
        bytes: mu-law encoded audio at 8kHz
    """
    import librosa
    # Resample to 8kHz if needed
    if sample_rate != 8000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=8000)
    # Normalize to [-1, 1]
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    max_val = np.abs(audio_data).max()
    if max_val > 0:
        audio_data = audio_data / max_val
    # Convert to 16-bit PCM
    pcm_data = (audio_data * 32767).astype(np.int16)
    # Convert to mu-law (simple approximation)
    # For production, use audioop.lin2ulaw
    import audioop
    mulaw_data = audioop.lin2ulaw(pcm_data.tobytes(), 2)
    return mulaw_data
 
 
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model': 'xtts_v2',
        'default_voice': DEFAULT_VOICE_SAMPLE if os.path.exists(DEFAULT_VOICE_SAMPLE) else None
    })
 
 
@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Generate speech from text
    Request:
        - JSON: {"text": "...", "language": "en"}
        - OR Form data with optional speaker_wav file
    Response:
        Audio file (WAV or mu-law depending on format parameter)
    """
    try:
        # Get text and language
        if request.is_json:
            data = request.json
            text = data.get('text')
            language = data.get('language', 'en')
            speaker_wav = DEFAULT_VOICE_SAMPLE if os.path.exists(DEFAULT_VOICE_SAMPLE) else None
        else:
            text = request.form.get('text')
            language = request.form.get('language', 'en')
            # Check for uploaded voice sample
            if 'speaker_wav' in request.files:
                file = request.files['speaker_wav']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    speaker_wav = filepath
                else:
                    speaker_wav = DEFAULT_VOICE_SAMPLE if os.path.exists(DEFAULT_VOICE_SAMPLE) else None
            else:
                speaker_wav = DEFAULT_VOICE_SAMPLE if os.path.exists(DEFAULT_VOICE_SAMPLE) else None
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        print(f"🎤 Generating speech: '{text[:50]}...'")
        print(f"   Language: {language}")
        print(f"   Voice: {speaker_wav if speaker_wav else 'default'}")
        # Generate speech
        if speaker_wav and os.path.exists(speaker_wav):
            # Voice cloning
            wav = tts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language
            )
        else:
            # Default voice
            wav = tts.tts(
                text=text,
                language=language
            )
        # Convert to numpy array
        wav_np = np.array(wav, dtype=np.float32)
        # Get output format
        output_format = request.args.get('format', 'wav')
        if output_format == 'mulaw':
            # Convert to mu-law 8kHz for Twilio
            mulaw_data = convert_to_mulaw_8khz(wav_np, 22050)
            # Return as bytes
            output = io.BytesIO(mulaw_data)
            output.seek(0)
            return send_file(
                output,
                mimetype='audio/basic',
                as_attachment=False
            )
        else:
            # Return as WAV
            output = io.BytesIO()
            sf.write(output, wav_np, 22050, format='WAV')
            output.seek(0)
            return send_file(
                output,
                mimetype='audio/wav',
                as_attachment=False,
                download_name='output.wav'
            )
    except Exception as e:
        print(f"❌ Error generating speech: {e}")
        return jsonify({'error': str(e)}), 500
 
 
@app.route('/upload-voice', methods=['POST'])
def upload_voice():
    """
    Upload a voice sample for cloning
    Request:
        Form data with 'voice_sample' file
    Response:
        JSON with file path
    """
    if 'voice_sample' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['voice_sample']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filepath': filepath,
            'message': 'Voice sample uploaded successfully'
        })
    return jsonify({'error': 'Invalid file type'}), 400
 
 
@app.route('/test', methods=['GET'])
def test():
    """
    Test endpoint - generates a simple test message
    """
    text = "Hello, this is a test message from the XTTS server."
    try:
        speaker_wav = DEFAULT_VOICE_SAMPLE if os.path.exists(DEFAULT_VOICE_SAMPLE) else None
        if speaker_wav:
            wav = tts.tts(text=text, speaker_wav=speaker_wav, language="en")
        else:
            wav = tts.tts(text=text, language="en")
        # Convert to WAV
        wav_np = np.array(wav, dtype=np.float32)
        output = io.BytesIO()
        sf.write(output, wav_np, 22050, format='WAV')
        output.seek(0)
        return send_file(
            output,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='test.wav'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
 
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤 XTTS TTS Server")
    print("="*60)
    print(f"📱 Device: {device}")
    print(f"🎵 Default voice: {DEFAULT_VOICE_SAMPLE if os.path.exists(DEFAULT_VOICE_SAMPLE) else 'None'}")
    print(f"🌐 Starting server on http://localhost:8000")
    print("="*60 + "\n")
    print("📚 API Endpoints:")
    print("   GET  /health          - Health check")
    print("   POST /tts             - Generate speech")
    print("   POST /upload-voice    - Upload voice sample")
    print("   GET  /test            - Test generation")
    print("")
    print("🔗 After starting, expose with ngrok:")
    print("   ngrok http 8000")
    print("")
    app.run(host='0.0.0.0', port=8000, debug=False)