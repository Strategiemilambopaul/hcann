import cv2
import threading
import time
from collections import deque

class VideoCapture:
    """Capture vidéo en temps réel avec buffer"""
    
    def __init__(self, source=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.frame = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        self.frame_buffer = deque(maxlen=30)  # Buffer de 30 frames
        
    def start(self):
        """Démarre la capture en thread séparé"""
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("📹 Webcam démarrée")
        
    def stop(self):
        """Arrête la capture"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        self.cap.release()
        
    def _capture_loop(self):
        """Boucle de capture continue"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Conversion BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frame = frame_rgb
                    self.frame_buffer.append({
                        'frame': frame_rgb,
                        'timestamp': time.time()
                    })
            time.sleep(1/30)  # 30 fps
            
    def get_frame(self):
        """Retourne la frame courante"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def get_recent_frames(self, seconds=2):
        """Retourne les frames des dernières secondes"""
        current_time = time.time()
        cutoff = current_time - seconds
        
        with self.lock:
            return [f for f in self.frame_buffer if f['timestamp'] >= cutoff]