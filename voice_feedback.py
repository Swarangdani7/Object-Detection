import os
import playsound
from gtts import gTTS

def start_sound(text):
    desc = ', '.join(text)
    filename = "voice.mp3"
    tts = gTTS(desc, lang="en")
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)
