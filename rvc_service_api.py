# services/rvc_service_api.py
import os
import time
import requests

# ----------------------------
# Load environment variables
# ----------------------------
RVC_API_URL = "https://api.replicate.com/v1/predictions"
RVC_MODEL = os.getenv("RVC_MODEL")  # e.g. "pseudoram/rvc-v2"
RVC_MODEL_VERSION = os.getenv("RVC_MODEL_VERSION")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REFERENCE_AUDIO = os.getenv("REFERENCE_AUDIO")


# ----------------------------
# Upload Helper
# ----------------------------
def upload_to_fileio(file_path):
    """Upload local file to transfer.sh for temporary hosting."""
    try:
        print(f"📁 Uploading with transfer.sh: {file_path}")
        if not file_path or not os.path.exists(file_path):
            print("❌ File not found or invalid path.")
            return None

        with open(file_path, "rb") as f:
            response = requests.put(f"https://transfer.sh/{os.path.basename(file_path)}", data=f)

        if response.status_code == 200:
            url = response.text.strip()
            print(f"☁️ Uploaded {file_path} → {url}")
            return url
        else:
            print(f"⚠️ transfer.sh upload failed: {response.status_code} → {response.text}")
            return None
    except Exception as e:
        print(f"❌ transfer.sh upload failed: {e}")
        return None


# ----------------------------
# Voice Cloning Function
# ----------------------------
def clone_voice(input_audio, output_audio="results/kalam_cloned.wav", reference_audio=None):
    """
    Clone voice using Replicate RVC API.
    Automatically uses the uploaded or .env reference audio.
    """
    try:
        os.makedirs("results", exist_ok=True)

        # Pick the right reference
        ref_audio = reference_audio or REFERENCE_AUDIO
        print("🎤 Sending audio to Replicate RVC API...")
        print(f"🧩 Using reference voice: {ref_audio}")

        ref_url = upload_to_fileio(ref_audio)
        tts_url = upload_to_fileio(input_audio)

        if not ref_url or not tts_url:
            print("❌ Could not upload audio files. Please check paths.")
            return None

        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "version": RVC_MODEL_VERSION,
            "input": {
                "model": RVC_MODEL,
                "input_audio": tts_url,
                "reference_audio": ref_url
            }
        }

        print(f"📡 Sending RVC clone request to {RVC_API_URL}...")
        response = requests.post(RVC_API_URL, headers=headers, json=payload)

        if response.status_code == 401:
            print("❌ Unauthorized: Invalid Replicate API token or access denied.")
            print(response.text)
            return None
        elif response.status_code not in [200, 201]:
            print(f"⚠️ RVC Error: {response.status_code}")
            print(response.text)
            return None

        prediction = response.json()
        prediction_url = prediction["urls"]["get"]

        print("⏳ Waiting for Replicate RVC model to finish...")
        while prediction["status"] not in ["succeeded", "failed", "canceled"]:
            time.sleep(5)
            prediction = requests.get(prediction_url, headers=headers).json()

        if prediction["status"] == "succeeded":
            output_url = prediction["output"][0]
            os.system(f"curl -L {output_url} -o {output_audio}")
            print(f"✅ Cloned Kalam voice saved: {output_audio}")
            return output_audio
        else:
            print(f"❌ RVC model failed: {prediction}")
            return None

    except Exception as e:
        print(f"❌ RVC API failed: {e}")
        return None
