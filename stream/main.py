#!/usr/bin/env python3
"""
HCANN Stream Launcher
Interface CLI/Web pour interagir avec l'agent en temps réel
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

# HCANN imports
from utils.config import HCANNConfig
from stream.agent import HCANN_Agent
from utils.entity_utils import EntityRegistry, merge_entities

# Web server (optionnel)
try:
    from fastapi import FastAPI, WebSocket, BackgroundTasks
    from fastapi.responses import HTMLResponse
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

# CLI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, level: str = "info"):
    """Affiche un message status coloré"""
    colors = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED
    }
    prefix = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✗"}.get(level, "•")
    print(f"{colors.get(level, '')}{prefix} {msg}{Colors.ENDC}")

async def cli_loop(agent: HCANN_Agent):
    """Boucle CLI interactive"""
    print_status("Mode CLI activé. Tapez vos questions ou 'quit' pour quitter.", "info")
    print_status("Commandes: /entities, /memory, /status, /help", "info")
    
    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}Vous:{Colors.ENDC} ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            # Commandes spéciales
            if user_input.startswith('/'):
                cmd = user_input.split()[0].lower()
                if cmd == '/entities':
                    entities = agent.entity_registry.list_entities()
                    print(f"\n{Colors.HEADER}📋 Entités connues:{Colors.ENDC}")
                    for eid, info in entities.items():
                        print(f"  • {eid}: {info.get('name', 'N/A')} | faces: {len(info.get('faces', []))}, voices: {len(info.get('voices', []))}")
                elif cmd == '/memory':
                    stats = agent.get_memory_stats()
                    print(f"\n{Colors.HEADER}📊 Statistiques mémoire:{Colors.ENDC}")
                    for k, v in stats.items():
                        print(f"  • {k}: {v}")
                elif cmd == '/status':
                    print(f"\n{Colors.HEADER}🔧 Status:{Colors.ENDC}")
                    print(f"  • Running: {agent.is_running}")
                    print(f"  • Frames processed: {agent.frame_count}")
                    print(f"  • Memory nodes: {len(agent.memory_graph.nodes)}")
                elif cmd == '/help':
                    print(f"\n{Colors.HEADER}❓ Commandes disponibles:{Colors.ENDC}")
                    print("  /entities  - Lister les entités connues")
                    print("  /memory    - Afficher les stats mémoire")
                    print("  /status    - Status de l'agent")
                    print("  /help      - Afficher cette aide")
                    print("  quit/exit  - Quitter")
                continue
                
            # Question normale
            start = time.time()
            response = agent.query_memory(user_input)
            latency = (time.time() - start) * 1000
            
            print(f"\n{Colors.GREEN}🤖 HCANN:{Colors.ENDC} {response}")
            print(f"{Colors.YELLOW}⏱️ Latence:{Colors.ENDC} {latency:.1f}ms")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print_status(f"Erreur: {e}", "error")

# ─────────────────────────────────────────────────────────────
# Interface Web (FastAPI) - Optionnelle
# ─────────────────────────────────────────────────────────────

if WEB_AVAILABLE:
    app = FastAPI(title="HCANN Stream API", version="2.0")
    _agent_instance: Optional[HCANN_Agent] = None
    
    @app.on_event("startup")
    async def startup():
        global _agent_instance
        config = HCANNConfig()
        _agent_instance = HCANN_Agent(config)
        _agent_instance.start()
        print_status("🌐 API Web démarrée sur http://localhost:8000", "success")
    
    @app.on_event("shutdown")
    async def shutdown():
        if _agent_instance:
            _agent_instance.stop()
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Interface web minimale"""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>HCANN Stream</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
            #chat { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; height: 400px; overflow-y: auto; }
            .msg { margin: 0.5rem 0; padding: 0.5rem; border-radius: 4px; }
            .user { background: #e3f2fd; margin-left: 2rem; }
            .agent { background: #f3e5f5; margin-right: 2rem; }
            #input { display: flex; gap: 0.5rem; margin-top: 1rem; }
            #question { flex: 1; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 0.5rem 1rem; background: #6200ea; color: white; border: none; border-radius: 4px; cursor: pointer; }
        </style></head>
        <body>
            <h1>🧠 HCANN Stream</h1>
            <div id="chat"></div>
            <div id="input">
                <input id="question" placeholder="Posez une question..." onkeypress="if(event.key==='Enter')send()">
                <button onclick="send()">Envoyer</button>
            </div>
            <script>
                const chat = document.getElementById('chat');
                function add(msg, cls) {
                    const div = document.createElement('div');
                    div.className = 'msg ' + cls;
                    div.textContent = msg;
                    chat.appendChild(div);
                    chat.scrollTop = chat.scrollHeight;
                }
                async function send() {
                    const input = document.getElementById('question');
                    const q = input.value.trim();
                    if (!q) return;
                    add(q, 'user');
                    input.value = '';
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: q})
                    });
                    const data = await res.json();
                    add(data.answer, 'agent');
                }
            </script>
        </body></html>
        """
    
    @app.post("/ask")
    async def ask_question(req: Dict[str, str]):
        if not _agent_instance:
            return {"answer": "Agent non initialisé"}
        answer = _agent_instance.query_memory(req["question"])
        return {"answer": answer}
    
    @app.get("/entities")
    async def list_entities():
        if not _agent_instance:
            return []
        return _agent_instance.entity_registry.list_entities()
    
    @app.get("/status")
    async def get_status():
        if not _agent_instance:
            return {"running": False}
        return {
            "running": _agent_instance.is_running,
            "frame_count": _agent_instance.frame_count,
            "memory_nodes": len(_agent_instance.memory_graph.nodes)
        }

def run_web(host: str = "0.0.0.0", port: int = 8000):
    """Lance le serveur web"""
    if not WEB_AVAILABLE:
        print_status("FastAPI non installé. Installez: pip install fastapi uvicorn", "error")
        return
    uvicorn.run("stream.main:app", host=host, port=port, reload=False)

# ─────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HCANN Stream Launcher")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", help="Mode d'interaction")
    parser.add_argument("--host", default="0.0.0.0", help="Hôte pour le serveur web")
    parser.add_argument("--port", type=int, default=8000, help="Port pour le serveur web")
    parser.add_argument("--video-source", type=str, default="0", help="Source vidéo (0 = webcam, ou chemin fichier)")
    parser.add_argument("--config", type=str, default=None, help="Chemin vers un fichier config JSON")
    
    args = parser.parse_args()
    
    # Chargement config
    config = HCANNConfig()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_dict = json.load(f)
            for k, v in config_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        print_status(f"Config chargée depuis {args.config}", "success")
    
    # Initialisation agent
    agent = HCANN_Agent(config, video_source=args.video_source)
    
    try:
        if args.mode == "cli":
            agent.start()
            asyncio.run(cli_loop(agent))
        elif args.mode == "web":
            # L'agent est démarré via l'événement startup de FastAPI
            run_web(args.host, args.port)
    finally:
        agent.stop()
        print_status("👋 Agent arrêté proprement", "info")

if __name__ == "__main__":
    main()