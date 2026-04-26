from stream.agent import HCANN_Agent
from utils.config import HCANNConfig
import time

def main():
    config = HCANNConfig()
    agent = HCANN_Agent(config)
    
    try:
        print("Agent démarré. Test de traitement...")
        
        # Test simple sans webcam
        import numpy as np
        for i in range(5):
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01
            agent.process_step(dummy_frame, dummy_audio)
            time.sleep(0.1)
        
        # Test de requête
        response = agent.query_memory("Qui était présent ?")
        print(response)
        
    except KeyboardInterrupt:
        print("\nArrêt de l'agent...")
    finally:
        agent.cleanup()

if __name__ == "__main__":
    main()