from stream.agent import HCANNStream
from utils.config import HCANNConfig
import time

def main():
    config = HCANNConfig()
    agent = HCANNStream(config)
    
    try:
        agent.start()
        print("Agent démarré. Appuyez sur Ctrl+C pour arrêter.")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nArrêt de l'agent...")
        agent.stop()

if __name__ == "__main__":
    main()