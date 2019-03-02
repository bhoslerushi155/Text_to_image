"""
Each Recognizer instance has seven methods for recognizing speech from an audio source using various APIs. These are:

recognize_bing(): Microsoft Bing Speech
recognize_google(): Google Web Speech API
recognize_google_cloud(): Google Cloud Speech - requires installation of the google-cloud-speech package
recognize_houndify(): Houndify by SoundHound
recognize_ibm(): IBM Speech to Text
recognize_sphinx(): CMU Sphinx - requires installing PocketSphinx
recognize_wit(): Wit.ai
Of the seven, only recognize_sphinx() works offline with the CMU Sphinx engine. The other six all require an internet connection.

A full discussion of the features and benefits of each API is beyond the scope of this tutorial. Since SpeechRecognition ships with a default API key for the Google Web Speech API, you can get started with it right away. For this reason, we’ll use the Web Speech API in this guide. The other six APIs all require authentication with either an API key or a username/password combination. For more information, consult the SpeechRecognition
"""
import speech_recognition as sr
# import PyAudio            #library needed to record audio from microphone

def run():
    r=sr.Recognizer()
    instance=sr.Microphone()
    with instance as source:
        print("say something")
        audio=r.listen(source)
    text=r.recognize_google(audio)
    print("you said : ")
    print(text)

if __name__=="__main__":
    run()
