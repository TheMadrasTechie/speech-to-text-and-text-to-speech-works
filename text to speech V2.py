import os

from google.cloud import texttospeech # outdated or incomplete comparing to v1
from google.cloud import texttospeech_v1
import time
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"chetan1.json"

# Instantiates a client
client = texttospeech_v1.TextToSpeechClient()
quote = '  நண்பா நண்பி எல்லாரும் வணக்கம் எப்படி இருக்கீங்க'
synthesis_input = texttospeech_v1.SynthesisInput(text=quote)
voice = texttospeech_v1.VoiceSelectionParams(name='ta-IN-Standard-A',language_code="ta-IN", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
audio_config = texttospeech_v1.AudioConfig(audio_encoding=texttospeech_v1.AudioEncoding.MP3)
response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
with open(r"output4.mp3", "wb") as out:    
    out.write(response.audio_content)
time.sleep((len(quote) *3)/100)
import play_audio
#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