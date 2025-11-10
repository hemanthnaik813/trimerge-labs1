import requests
import os

def clone_voice(input_audio_path, output_audio_path="kalam_cloned.wav"):
    """
    Send TTS audio to RVC API to clone it into Kalam's voice.
    """

    RVC_API_URL = os.getenv("RVC_API_URL")

    if not RVC_API_URL:
        raise ValueError("❌ RVC_API_URL not found in .env file. Please add it.")

    try:
        files = {'file': open(input_audio_path, 'rb')}
        response = requests.post(f"{RVC_API_URL}/clone", files=files)

        if response.status_code == 200:
            with open(output_audio_path, "wb") as f:
                f.write(response.content)
            print(f"🎤 Cloned voice saved to: {output_audio_path}")
            return output_audio_path
        else:
            print(f"⚠️ RVC API Error: {response.status_code} {response.text}")
            return None
    except Exception as e:
        print(f"❌ Voice cloning failed: {e}")
        return None
