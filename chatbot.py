from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Pinecone
pc = Pinecone(api_key="pcsk_2fERb7_PUnzAn8cRe52YB5DbGMhL7wpzzRi8zzoW3H8dq7idGhVYza78fiBtAE7Uk3BTxD")
index = pc.Index("printer-chatbot")

while True:
    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # FILTER RESULTS (IMPORTANT FIX)
    matches = [m for m in results["matches"] if m["score"] > 0.75]

    print("\nAI Tutor Answer:\n")

    if not matches:
        print("No relevant answer found. Try rephrasing your question.")
    else:
        for m in matches:
            print("•", m["metadata"]["text"])
            print(f"(Score: {m['score']:.2f})")