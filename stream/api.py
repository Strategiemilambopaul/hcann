from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="HCANN Stream API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class AgentInstance:
    def __init__(self):
        self.agent = None

instance = AgentInstance()

@app.on_event("startup")
def startup():
    """Initialise l'agent au démarrage"""
    from utils.config import HCANNConfig
    from .agent import HCANNStream
    
    config = HCANNConfig()
    instance.agent = HCANNStream(config)
    instance.agent.start()
    print("✅ API démarrée")

@app.post("/ask")
async def ask_question(req: QueryRequest):
    """Pose une question à l'agent"""
    if instance.agent:
        answer = instance.agent.query(req.question)
        return {"answer": answer}
    return {"answer": "Agent non prêt"}

@app.get("/status")
async def get_status():
    """Statut de l'agent"""
    return {
        "status": "running" if instance.agent and instance.agent.is_running else "stopped",
        "buffer_size": len(instance.agent.clip_buffer) if instance.agent else 0
    }