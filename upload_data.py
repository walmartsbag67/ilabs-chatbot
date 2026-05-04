import pinecone
from sentence_transformers import SentenceTransformer

pc = pinecone.Pinecone(api_key="pcsk_2fERb7_PUnzAn8cRe52YB5DbGMhL7wpzzRi8zzoW3H8dq7idGhVYza78fiBtAE7Uk3BTxD")
index = pc.Index("printer-chatbot")

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("model.md", "r", encoding="utf-8") as f:
    text = f.read()

# IMPORTANT: split into chunks
chunks = text.split(". ")

vectors = []

for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    vectors.append((f"doc-{i}", embedding, {"text": chunk}))

index.upsert(vectors)

print("Data uploaded in chunks!")