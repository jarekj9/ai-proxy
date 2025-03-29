from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from azure_client import AzureOpenAIClient

app = FastAPI(title="AI Proxy Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Azure OpenAI client
try:
    client = AzureOpenAIClient()
except Exception as e:
    print(f"Warning: Failed to initialize Azure OpenAI client: {str(e)}")
    client = None

@app.get("/")
async def root():
    return {"message": "AI Proxy Service is running"}

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: dict):
    if not client:
        raise HTTPException(status_code=500, detail="Azure OpenAI client is not initialized")
    
    try:
        return client.complete(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 