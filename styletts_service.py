def synthesize_with_controls(model, text, output_path="results/kalam_styled.wav", reference_audio=None, pitch_control=0.7, energy_control=0.85, duration_control=1.25):
    try:
        # If we have a model, try to use it (with ref if provided) else just post-process later
        if model is not None and hasattr(model, "tts"):
            try:
                wav, sr = model.tts(
                    text,
                    style_vector=None,
                    pitch_control=pitch_control,
                    energy_control=energy_control,
                    duration_control=duration_control,
                )
                sf.write(output_path, np.asarray(wav), int(sr))
                return output_path
            except Exception:
                pass
        # Fallback: try PyPI inference (can use reference) then apply controls
        if model is not None:
            try:
                from styletts2 import tts as _tts
                audio = model.inference(
                    text,
                    target_voice_path=reference_audio,
                    output_wav_file=output_path,
                    output_sample_rate=24000,
                )
                if audio is not None and not os.path.exists(output_path):
                    sf.write(output_path, np.asarray(audio), 24000)
            except Exception:
                pass
        # If no file yet, just use gTTS or caller-provided base; here we only post-process if path exists
        if os.path.exists(output_path):
            adjust_voice_male(output_path, pitch_control=pitch_control, energy_control=energy_control, duration_control=duration_control)
            return output_path
        return None
    except Exception:
        return None
def _lowpass_sos(sr, cutoff=3800, order=4):
    nyq = 0.5 * sr
    norm = min(cutoff / nyq, 0.99)
    return butter(order, norm, btype='low', output='sos')

def _highpass_sos(sr, cutoff=80, order=4):
    nyq = 0.5 * sr
    norm = max(cutoff / nyq, 1e-4)
    return butter(order, norm, btype='high', output='sos')

def adjust_voice_male(wav_path, pitch_control=0.6, energy_control=0.9, duration_control=1.3):
    try:
        if not os.path.exists(wav_path):
            return None
        # Load WAV to float32 [-1,1]
        y, sr = sf.read(wav_path, dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Pitch shift in semitones (0.8 ~ -3.17 st, less aggressive for clearer male voice)
        n_steps = 12.0 * np.log2(max(1e-3, float(pitch_control)))
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        # Time-stretch so overall duration matches requested (duration_control > 1 => slower)
        stretch = 1.0 / float(duration_control)
        stretch = min(2.0, max(0.25, stretch))
        y = librosa.effects.time_stretch(y, rate=stretch)
        # Gentle EQ: high-pass 80 Hz, low-pass 4 kHz for clearer speech
        try:
            y = sosfilt(_highpass_sos(sr, 80), y)
            y = sosfilt(_lowpass_sos(sr, 4000), y)
        except Exception:
            pass
        # Energy control
        y = y * float(energy_control)
        # Normalize/clamp to avoid clipping, amplify to full volume for audibility
        peak = np.max(np.abs(y)) if y.size else 0.0
        if peak > 0:
            y = y / peak  # Normalize to [-1, 1]
            y = y * 0.98  # Scale to 98% to prevent clipping
        y = np.clip(y, -1.0, 1.0).astype('float32')
        # Save back (keep original sample rate)
        sf.write(wav_path, y, sr)
        print(f"🎚️ Applied male voice controls (pitch={pitch_control}, energy={energy_control}, duration={duration_control}).")
        return wav_path
    except Exception as e:
        print(f"⚠️ Post-processing failed: {e}")
        return None
import os
import torch
import soundfile as sf
import numpy as np
import librosa
from scipy.signal import butter, sosfilt
# Load the StyleTTS2 model (CPU mode)
def load_styletts():
    try:
        # Import lazily so the app can still run if package isn't installed
        from styletts2 import tts
        print("🎙️ Loading StyleTTS2 voice cloning model on CPU (PyPI package)...")
        model = tts.StyleTTS2()  # downloads/caches default checkpoints
        print("✅ StyleTTS2 loaded successfully (CPU mode).")
        return model
    except Exception as e:
        print(f"❌ Failed to load StyleTTS2: {e}")
        return None

# Clone voice using the reference sample (with optional voice controls)
def clone_voice(model, text, reference_audio, output_path="results/kalam_cloned.wav", pitch_control=0.7, energy_control=0.85, duration_control=1.25):
    try:
        if model is None:
            print("⚠️ StyleTTS2 model not available, skipping cloning.")
            return None
        print(f"🧩 Cloning with reference voice: {reference_audio}")
        if not reference_audio or not os.path.exists(reference_audio):
            print("⚠️ Reference voice not found, skipping cloning.")
            return None

        # Prefer native tts() API if available (research repo style)
        if hasattr(model, "tts"):
            try:
                wav, sr = model.tts(
                    text,
                    style_vector=None,
                    pitch_control=pitch_control,
                    energy_control=energy_control,
                    duration_control=duration_control,
                )
                sf.write(output_path, np.asarray(wav), int(sr))
                print(f"✅ Cloned Kalam voice saved: {output_path}")
                return output_path
            except Exception as _:
                pass

        # Fallback to PyPI inference + post processing
        audio = None
        try:
            from styletts2 import tts as _tts  # ensure API present
            audio = model.inference(
                text,
                target_voice_path=reference_audio,
                output_wav_file=output_path,
                output_sample_rate=24000,
            )
        except Exception:
            pass

        if audio is not None and not os.path.exists(output_path):
            sf.write(output_path, np.asarray(audio), 24000)

        if os.path.exists(output_path):
            adjust_voice_male(output_path, pitch_control=pitch_control, energy_control=energy_control, duration_control=duration_control)
            print(f"✅ Cloned Kalam voice saved: {output_path}")
            return output_path
        else:
            print("⚠️ Cloning did not produce an audio file.")
            return None
    except Exception as e:
        print(f"❌ Voice cloning failed: {e}")
        return None
