import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TTS_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

def synthesize_speech(text, output_path="output.wav"):
    # Use gTTS for natural human-like voice
    try:
        from gtts import gTTS
        import tempfile
        import librosa
        import soundfile as sf
        import numpy as np

        # Create gTTS with natural settings
        tts = gTTS(text=text, lang='en', slow=False, tld='com.au')  # Australian accent for natural male voice

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmpf:
            tmp_mp3 = tmpf.name
        tts.save(tmp_mp3)

        # Load and convert to WAV with high quality
        y, sr = librosa.load(tmp_mp3, sr=24000, mono=True)

        # Normalize to prevent clipping and amplify for audibility
        peak = np.max(np.abs(y)) if y.size else 0.0
        if peak > 0:
            y = y / peak  # Normalize to [-1, 1]
            y = y * 0.95  # Scale to 95% to prevent clipping, maximize volume

        sf.write(output_path, y, 24000)
        os.remove(tmp_mp3)
        print(f"🔊 Natural TTS audio generated: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ gTTS failed: {e}")
        # Fallback to pyttsx3
        try:
            import pyttsx3
            import tempfile
            import librosa
            import soundfile as sf
            import numpy as np

            engine = pyttsx3.init(driverName='sapi5')
            # Select male voice
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
            # Set volume to max
            engine.setProperty('volume', 1.0)
            # Slower rate
            rate = engine.getProperty('rate')
            engine.setProperty('rate', int(rate * 0.95))

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
                tmp_wav = tmpf.name
            engine.save_to_file(text, tmp_wav)
            engine.runAndWait()

            y, sr = sf.read(tmp_wav, dtype='float32', always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != 24000:
                y = librosa.resample(y, orig_sr=sr, target_sr=24000)
                sr = 24000
            sf.write(output_path, y, sr)
            os.remove(tmp_wav)
            print(f"🔊 Base TTS audio generated: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            return None
