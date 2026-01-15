#!/bin/bash

# ==================== GPU Voice Agent Deployment Script ====================
# This script automates the complete setup on a GPU instance

set -e  # Exit on error

echo "=========================================="
echo "🚀 GPU Voice Agent Deployment"
echo "=========================================="
echo ""

# ==================== 1. CHECK GPU ====================
echo "📊 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. GPU drivers not installed!"
    exit 1
fi

nvidia-smi
echo ""
echo "✅ GPU detected"
echo ""

# ==================== 2. INSTALL SYSTEM DEPENDENCIES ====================
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    portaudio19-dev \
    build-essential \
    wget

echo "✅ System dependencies installed"
echo ""

# ==================== 3. CREATE VIRTUAL ENVIRONMENT ====================
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "✅ Virtual environment created"
echo ""

# ==================== 4. INSTALL PYTHON PACKAGES ====================
echo "📦 Installing Python packages (this may take 10-15 minutes)..."

# Install PyTorch with CUDA
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core packages
pip install \
    flask==3.0.0 \
    twilio==8.11.0 \
    python-dotenv==1.0.0 \
    openai==1.3.0 \
    websockets==12.0

# Install GPU-accelerated ML packages
pip install \
    faster-whisper==0.10.0 \
    TTS==0.22.0 \
    sentence-transformers==2.2.2 \
    faiss-gpu==1.7.2 \
    fuzzywuzzy==0.18.0 \
    python-Levenshtein==0.23.0

# Install audio processing
pip install \
    numpy==1.24.3 \
    soundfile==0.12.1 \
    librosa==0.10.1

echo "✅ Python packages installed"
echo ""

# ==================== 5. VERIFY GPU SETUP ====================
echo "🔍 Verifying GPU setup..."

python3 << EOF
import torch
import faiss

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

print("\nFAISS GPU count:", faiss.get_num_gpus())
print("FAISS GPU available:", faiss.get_num_gpus() > 0)
EOF

echo ""
echo "✅ GPU verification complete"
echo ""

# ==================== 6. SETUP PROJECT STRUCTURE ====================
echo "📁 Setting up project structure..."

mkdir -p call_logs
mkdir -p models
mkdir -p voice_samples

echo "✅ Project structure created"
echo ""

# ==================== 7. DOWNLOAD PRE-TRAINED MODELS ====================
echo "📥 Downloading pre-trained models (this may take 5-10 minutes)..."

# Download Faster Whisper model
python3 << EOF
from faster_whisper import WhisperModel
print("Downloading Whisper large-v3...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("✅ Whisper model downloaded")
EOF

# Download Coqui TTS model
python3 << EOF
from TTS.api import TTS
print("Downloading Coqui TTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("✅ TTS model downloaded")
EOF

echo "✅ Models downloaded"
echo ""

# ==================== 8. BUILD RAG INDEX ====================
echo "🔨 Building RAG index..."

if [ -f "docs.txt" ]; then
    python3 << EOF
from gpu_rag_system import build_gpu_index
build_gpu_index()
EOF
    echo "✅ RAG index built"
else
    echo "⚠️ docs.txt not found - skipping RAG index build"
    echo "   Create docs.txt and run: python gpu_rag_system.py"
fi

echo ""

# ==================== 9. SETUP ENVIRONMENT ====================
echo "⚙️ Setting up environment variables..."

if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
PUBLIC_URL=https://your-server.com

# LLM Configuration
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_openai_key
LLM_MODEL=gpt-4

# Flask Configuration
FLASK_PORT=5000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_DEVICE=cuda

# Voice Cloning (optional)
VOICE_SAMPLE_PATH=/app/voice_sample.wav
EOF
    echo "⚠️ .env file created - PLEASE UPDATE WITH YOUR CREDENTIALS"
else
    echo "✅ .env file already exists"
fi

echo ""

# ==================== 10. CREATE SYSTEMD SERVICE ====================
echo "🔧 Creating systemd service..."

sudo tee /etc/systemd/system/medicaid-voice-agent.service > /dev/null << EOF
[Unit]
Description=Medicaid GPU Voice Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "✅ Systemd service created"
echo ""

# ==================== 11. CREATE MONITORING SCRIPT ====================
echo "📊 Creating monitoring script..."

cat > monitor.sh << 'EOF'
#!/bin/bash
# Monitor GPU usage and service status

echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

echo ""
echo "=== Service Status ==="
sudo systemctl status medicaid-voice-agent --no-pager | head -15

echo ""
echo "=== Active Calls ==="
curl -s http://localhost:5000/health | python3 -m json.tool

echo ""
echo "=== Recent Logs ==="
tail -20 voice_agent.log 2>/dev/null || echo "No logs yet"
EOF

chmod +x monitor.sh
echo "✅ Monitoring script created (run: ./monitor.sh)"
echo ""

# ==================== 12. CREATE TEST SCRIPT ====================
echo "🧪 Creating test script..."

cat > test_gpu.sh << 'EOF'
#!/bin/bash
# Test GPU components

echo "Testing GPU Voice Agent Components..."
echo ""

source venv/bin/activate

echo "1. Testing Faster Whisper (GPU STT)..."
python3 << PYEOF
import time
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("✅ Whisper model loaded on GPU")

# Create test audio (silence)
import numpy as np
test_audio = np.zeros(16000, dtype=np.float32)

start = time.time()
segments, _ = model.transcribe(test_audio)
_ = list(segments)
elapsed = time.time() - start
print(f"✅ Transcription test: {elapsed:.3f}s")
PYEOF

echo ""
echo "2. Testing Coqui TTS (GPU voice synthesis)..."
python3 << PYEOF
import time
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("✅ TTS model loaded on GPU")

text = "This is a test message."
start = time.time()
wav = tts.tts(text=text, language="en")
elapsed = time.time() - start
print(f"✅ TTS synthesis test: {elapsed:.3f}s")
PYEOF

echo ""
echo "3. Testing FAISS-GPU (RAG search)..."
python3 << PYEOF
import time
import faiss
import numpy as np

print(f"FAISS GPU count: {faiss.get_num_gpus()}")

# Create test index
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Add random vectors
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

# Move to GPU if available
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    print("✅ FAISS index moved to GPU")
    
    # Search
    query = np.random.random((1, dimension)).astype('float32')
    start = time.time()
    for _ in range(100):
        distances, indices = gpu_index.search(query, 5)
    elapsed = time.time() - start
    print(f"✅ 100 searches: {elapsed:.3f}s ({elapsed/100*1000:.1f}ms per search)")
else:
    print("⚠️ No GPU available for FAISS")
PYEOF

echo ""
echo "✅ All GPU components tested successfully!"
EOF

chmod +x test_gpu.sh
echo "✅ Test script created (run: ./test_gpu.sh)"
echo ""

# ==================== 13. CREATE BACKUP SCRIPT ====================
echo "💾 Creating backup script..."

cat > backup.sh << 'EOF'
#!/bin/bash
# Backup call logs and configurations

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup in $BACKUP_DIR..."

# Backup call logs
cp -r call_logs "$BACKUP_DIR/" 2>/dev/null
echo "✅ Call logs backed up"

# Backup configuration
cp .env "$BACKUP_DIR/" 2>/dev/null
echo "✅ Configuration backed up"

# Backup RAG index
cp docs.index "$BACKUP_DIR/" 2>/dev/null
cp docs.pkl "$BACKUP_DIR/" 2>/dev/null
echo "✅ RAG index backed up"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "✅ Backup complete: $BACKUP_DIR.tar.gz"
EOF

chmod +x backup.sh
echo "✅ Backup script created (run: ./backup.sh)"
echo ""

# ==================== 14. FINAL INSTRUCTIONS ====================
echo ""
echo "=========================================="
echo "✅ GPU Voice Agent Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Update .env file with your credentials:"
echo "   nano .env"
echo ""
echo "2. Add your voice sample (6-30 seconds):"
echo "   cp your_voice.wav voice_sample.wav"
echo ""
echo "3. Test GPU components:"
echo "   ./test_gpu.sh"
echo ""
echo "4. Start the service:"
echo "   sudo systemctl start medicaid-voice-agent"
echo "   sudo systemctl enable medicaid-voice-agent"
echo ""
echo "5. Monitor the service:"
echo "   ./monitor.sh"
echo ""
echo "6. Check logs:"
echo "   tail -f voice_agent.log"
echo ""
echo "📊 Useful Commands:"
echo "   - Service status:  sudo systemctl status medicaid-voice-agent"
echo "   - Restart:         sudo systemctl restart medicaid-voice-agent"
echo "   - Stop:            sudo systemctl stop medicaid-voice-agent"
echo "   - Logs:            journalctl -u medicaid-voice-agent -f"
echo "   - GPU monitor:     watch -n 1 nvidia-smi"
echo "   - Backup:          ./backup.sh"
echo ""
echo "🔗 Endpoints:"
echo "   - Health:          http://localhost:5000/health"
echo "   - Make call:       http://localhost:5000/make-call"
echo ""
echo "=========================================="