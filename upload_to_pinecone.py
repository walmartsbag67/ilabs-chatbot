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
# Added http_options to specify v1beta so it can find text-embedding-004
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={'api_version': 'v1beta'}
)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 3. LOAD DATA
if not os.path.exists("Model.md"):
    print("ERROR: Model.md not found!")
    exit()

with open("Model.md", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks (using double newlines as a separator)
chunks = [c for c in text.split("\n\n") if c.strip()]

# 4. EMBED AND UPLOAD
print(f"Uploading {len(chunks)} sections to Pinecone...")

for i, chunk in enumerate(chunks):
    try:
        print(f"Embedding section {i+1}...")
        
        # 1. Generate the embedding (768 dimensions)
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=chunk
        )
        vector = result.embeddings[0].values

        # 2. Upload to Pinecone
        # Ensure metadata key matches what your app.py expects (usually "text")
        index.upsert(vectors=[(f"id-{i}-{uuid.uuid4().hex[:6]}", vector, {"text": chunk})])
        
    except Exception as e:
        print(f"Failed to upload section {i+1}: {e}")

print("SUCCESS: Your knowledge is now live in the cloud!")