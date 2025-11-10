from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from services.styletts_service import load_styletts, clone_voice, adjust_voice_male  # ✅ Local voice cloning

import librosa
import soundfile as sf
import numpy as np
import tempfile
import pyttsx3

# ----------------------------
# ✅ Setup
# ----------------------------
load_dotenv()

os.makedirs("results", exist_ok=True)
os.makedirs("samples", exist_ok=True)

app = Flask(__name__)

MODEL_DIR = "models/kalam_brain/merged_model"

print(f"🧠 Loading Kalam Brain from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float32,  # ✅ CPU only (replaces deprecated torch_dtype)
    device_map=None,
    low_cpu_mem_usage=True
)
model.eval()
print("✅ Kalam Brain Loaded Successfully!")

# ✅ Load StyleTTS2 (Local Voice Cloning)
styletts_model = load_styletts()


# ----------------------------
# 🔊 Text → Base Speech (TTS)
# ----------------------------
def synthesize_speech(text, output_file="results/kalam_tts.wav"):
    """Offline base TTS using pyttsx3 to WAV, then resample to 24kHz mono. No ffmpeg required."""
    try:
        # Use explicit SAPI5 driver on Windows
        engine = pyttsx3.init(driverName='sapi5')
        # Try to select a male voice if available
        try:
            voices = engine.getProperty('voices')
            male_voice_id = None
            for v in voices:
                name = (getattr(v, 'name', '') or '').lower()
                gender = (getattr(v, 'gender', '') or '').lower()
                if 'male' in gender or 'male' in name or 'david' in name or 'george' in name:
                    male_voice_id = v.id
                    break
            if male_voice_id:
                engine.setProperty('voice', male_voice_id)
        except Exception:
            pass
        # Slightly slower for base voice (we still post-process later)
        try:
            rate = engine.getProperty('rate')
            engine.setProperty('rate', int(rate * 0.95))
        except Exception:
            pass
        # Maximize volume for audibility
        try:
            volume = engine.getProperty('volume')
            engine.setProperty('volume', 1.0)
        except Exception:
            pass
        # Save to a temporary WAV via pyttsx3
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
            tmp_wav = tmpf.name
        # IMPORTANT: pyttsx3 uses snake_case
        engine.save_to_file(text, tmp_wav)
        engine.runAndWait()
        # Load, convert to mono 24kHz, write final WAV
        y, sr = sf.read(tmp_wav, dtype='float32', always_2d=False)
        if hasattr(y, 'ndim') and y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 24000:
            y = librosa.resample(y, orig_sr=sr, target_sr=24000)
            sr = 24000
        sf.write(output_file, y, sr)
        try:
            os.remove(tmp_wav)
        except Exception:
            pass
        print(f"🔊 Base TTS audio generated: {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        return None


# ----------------------------
# 🎙️ Upload Reference Voice
# ----------------------------
UPLOAD_FOLDER = "samples"

@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    """Upload a reference voice file (e.g., kalam_reference.wav)."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    print(f"✅ Uploaded new reference voice: {save_path}")
    return jsonify({"message": "Voice uploaded successfully!", "path": save_path})


# ----------------------------
# 🧠 Chat Endpoint
# ----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("text", "").strip()
    reference_audio = data.get("reference_audio", "samples/kalam_reference.wav")

    if not user_text:
        return jsonify({"error": "text is required"}), 400

    print(f"👤 User: {user_text}")

    # 🧠 Generate Kalam-style response
    inputs = tokenizer(user_text, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.8,
        repetition_penalty=1.2,
        do_sample=True
    )

    kalam_response = tokenizer.decode(output[0], skip_special_tokens=True)
    if user_text in kalam_response:
        kalam_response = kalam_response.split(user_text, 1)[-1].strip()

    if not kalam_response or len(kalam_response) < 5:
        kalam_response = "My dear students, always dream big and work hard to achieve greatness."

    print(f"🧠 Kalam: {kalam_response}")

    # Step 1: Base TTS - Use pyttsx3 for now, Google TTS needs API setup
    base_audio = synthesize_speech(kalam_response, "results/kalam_tts.wav")
    if not base_audio:
        return jsonify({"error": "Failed to generate TTS audio"}), 500

    # Desired male/older voice controls
    PITCH = 0.8
    ENERGY = 1.0
    DURATION = 1.0

    # Step 2: Clone Voice Locally with controls
    cloned_audio = clone_voice(
        styletts_model,
        kalam_response,
        reference_audio,
        "results/kalam_cloned.wav",
        pitch_control=PITCH,
        energy_control=ENERGY,
        duration_control=DURATION,
    )

    # Use base audio directly for natural human-like voice
    final_audio = base_audio

    # Optionally also export a copy with the requested name
    try:
        if final_audio and os.path.exists(final_audio):
            import shutil
            shutil.copyfile(final_audio, "results/output_kalam_style.wav")
    except Exception:
        pass

    return jsonify({
        "response": kalam_response,
        "audio_file": final_audio
    })


# ----------------------------
# 🚀 Run Flask
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting APJ Flask Server (CPU Mode)...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
