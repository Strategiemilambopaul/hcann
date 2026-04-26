import pyaudio
import whisper
import threading
import time
import numpy as np
from collections import deque

class AudioCapture:
    """Capture audio avec transcription temps réel"""
    
    def __init__(self, chunk_size=4096, sample_rate=16000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.thread = None
        
        # Buffer audio
        self.audio_buffer = deque(maxlen=10)
        
        # Whisper pour transcription
        print("🎙️ Chargement du modèle Whisper...")
        self.whisper = whisper.load_model("base")
        print("✅ Whisper prêt")
        
        # Transcriptions récentes
        self.transcriptions = deque(maxlen=20)
        
    def start(self):
        """Démarre la capture audio"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        self.is_running = True
        self.thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.thread.start()
        print("🎙️ Audio démarré")
        
    def stop(self):
        """Arrête la capture"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        
    def _transcribe_loop(self):
        """Boucle de transcription continue"""
        while self.is_running:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Transcription
                result = self.whisper.transcribe(
                    audio_array,
                    language='fr',
                    fp16=False
                )
                
                if result['text'].strip():
                    transcription = {
                        'text': result['text'].strip(),
                        'timestamp': time.time(),
                        'confidence': result.get('segments', [{}])[0].get('avg_logprob', 0)
                    }
                    self.transcriptions.append(transcription)
                    print(f"🗣️ {transcription['text']}")
                    
            except Exception as e:
                print(f"Erreur transcription: {e}")
                
    def get_recent_transcriptions(self, seconds=10):
        """Récupère les transcriptions récentes"""
        current_time = time.time()
        cutoff = current_time - seconds
        
        return [t for t in self.transcriptions if t['timestamp'] >= cutoff]