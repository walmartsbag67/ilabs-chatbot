import os
import json
from google import genai
from google.genai import types
from google.oauth2 import service_account
from pinecone import Pinecone

# --- 1. CONFIGURATION ---
GCP_KEY_FILE = "makerspace-production-bot-7abdbb3eada1.json" 
PINECONE_API_KEY = "pcsk_5xzpBL_3XaaGKyPZCrwBWmbBZSvixCZqm1jk4fvH421rh994R4ejQhr5kZ84EtVbRYsrdr"  # 🔑 Replace with your real Pinecone API key
PINECONE_INDEX_NAME = "printer-bot"            # Matches your Pinecone index name
DATA_FILE = "model.md"                             # Your lowercase markdown database file

# --- 2. AUTHENTICATE VIA VERTEX AI (GOOGLE CLOUD) ---
print("Configuring Google Cloud credentials...")
if not os.path.exists(GCP_KEY_FILE):
    raise FileNotFoundError(f"Missing service account file: {GCP_KEY_FILE} in your root folder.")

with open(GCP_KEY_FILE, 'r') as f:
    sa_info = json.load(f)

# Bind explicit cloud platform scopes to authorize requests cleanly
credentials = service_account.Credentials.from_service_account_info(sa_info).with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform']
)

# Initialize Client pointing directly to your new project space container
client = genai.Client(
    vertexai=True,
    project=sa_info["project_id"],
    location="asia-southeast1",
    credentials=credentials
)

# --- 3. INITIALIZE PINECONE ---
print("Connecting to Pinecone Database...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- 4. HELPER FUNCTIONS ---
def get_chunk_embedding(text_chunk: str) -> list:
    """Generates 768-dimension vectors using Google's text-embedding-004."""
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text_chunk,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT" # Optimizes the vector structure for index storage
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding Generation Error on chunk: {e}")
        return None

def chunk_markdown(file_path: str, chunk_size: int = 500) -> list:
    """Splits your model.md file into cleanly separated structural paragraphs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find source file: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

# --- 5. EXECUTION PIPELINE ---
def main():
    # Step A: Document Parsing
    print(f"Splitting '{DATA_FILE}' into structural text blocks...")
    chunks = chunk_markdown(DATA_FILE)
    print(f"Successfully split into {len(chunks)} text chunks.")
    
    # Step B: Clean the target space
    print("Clearing out any old mismatched developer vectors from the index...")
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Notice (Safe to ignore if index was already empty): {e}")

    # Step C: Generate Embeddings & Compile Payload
    print("Requesting 768-dimension vectors from Google Cloud Vertex AI...")
    upsert_data = []
    
    for idx, chunk_text in enumerate(chunks):
        print(f"Vectorizing chunk {idx + 1}/{len(chunks)}...")
        
        # This calls Google's text-embedding-004 model directly
        vector = get_chunk_embedding(chunk_text)
        
        if vector:
            upsert_data.append({
                "id": f"chunk_{idx}",
                "values": vector,
                "metadata": {"text": chunk_text} # Keeps metadata key mapped directly to app.py query requirements
            })
            
    # Step D: Bulk Upsert Data to Pinecone
    if upsert_data:
        print(f"Uploading vectors into Pinecone index '{PINECONE_INDEX_NAME}'...")
        # explicitly passing vectors as a keyword argument to prevent the TypeError
        index.upsert(vectors=upsert_data)
        print("🎉 Successfully uploaded! Your database and app are now completely synchronized.")
    else:
        print("❌ Data vector payload generation failed completely.")

if __name__ == "__main__":
    main()