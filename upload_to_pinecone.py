import os
import uuid
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

# 1. SETUP
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "printer-bot"

# 2. INITIALIZE CLIENTS
# Using the modern 'google-genai' client
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 3. LOAD DATA
if not os.path.exists("Model.md"):
    print("ERROR: Model.md not found!")
    exit()

with open("Model.md", "r", encoding="utf-8") as f:
    text = f.read()
chunks = [c for c in text.split("\n\n") if c.strip()]

# 4. EMBED AND UPLOAD
print(f"Uploading {len(chunks)} sections to Pinecone...")

for i, chunk in enumerate(chunks):
    print(f"Embedding section {i+1}...")
    
    # FORCING API v1 to avoid the 404 error
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=chunk,
        config={'api_version'} 
    )
    
    # Send to Pinecone
    index.upsert(vectors=[{
        "id": str(uuid.uuid4()),
        "values": result.embeddings[0].values,
        "metadata": {"text": chunk}
    }])

print("SUCCESS: Your knowledge is now live in the cloud!")