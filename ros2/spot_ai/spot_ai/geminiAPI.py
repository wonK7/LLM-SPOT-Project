import rclpy
from rclpy.node import Node
import google.generativeai as genai

import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

from gtts import gTTS
import os

from playsound import playsound


class SpotAIBrain(Node):
    def __init__(self):
        super().__init__('geminiAPI_node')
        
        # 1. Gemini Configuration
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key, transport='rest')
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Automatically select the first available model 
            self.model_name = available_models[0]
            self.model = genai.GenerativeModel(self.model_name)
            self.get_logger().info(f'Using model: {self.model_name}')
        except Exception as e:
            self.get_logger().error(f'Could not find any available models: {e}')
            return

        self.get_logger().info('🎤 Speak something to Spot...')

         # 2️. Record voice and transcribe
        voice_text = self.record_and_transcribe()

        print(f"\n[Recognized Text]: {voice_text}")

        # 3️. Send to Gemini
        self.call_gemini(voice_text)

    # def record_and_transcribe(self, duration=5, fs=16000):
    #     print("🎙 Recording...")
    #     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    #     sd.wait()

    #     write("input.wav", fs, recording)

    #     self.get_logger().info("Transcribing...")

    #     model = whisper.load_model("base")
    #     result = model.transcribe("input.wav")

    #     return result["text"]

    def record_and_transcribe(self):
        self.get_logger().info("Reading audio file from Windows...")

        model = whisper.load_model("base")

        # Windows Desktop path
        audio_path = "/mnt/c/Users/jaeyk/Desktop/input.wav"

        result = model.transcribe(audio_path)

        return result["text"]
        
        #self.get_logger().info('Spot AI Brain is now online and thinking...')
        
        # Initial greeting and test call to Gemini
        #self.call_gemini("Hello, Spot! I'm your new friend, Hyewon. "
        #    "Please introduce yourself briefly and can you tell me a joke?")

    def call_gemini(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            text = response.text

            print(f"\n[Spot's response]:\n{text}")

            # voice output
            tts = gTTS(text)
            tts.save("speech.mp3")
            os.system("mpg123 speech.mp3")
            #playsound("speech.mp3")

        except Exception as e:
            self.get_logger().error(f"Error calling the brain: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SpotAIBrain()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
