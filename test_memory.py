from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

# 1. Load the "Translator" (Embedding Model)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Connect to Pinecone
# Replace 'YOUR_API_KEY' with your actual Pinecone key
pc = Pinecone(api_key="pcsk_2fERb7_PUnzAn8cRe52YB5DbGMhL7wpzzRi8zzoW3H8dq7idGhVYza78fiBtAE7Uk3BTxD")
index = pc.Index("printer-dashboard") # Replace with your index name

# 3. Try to "Search" for something
test_query = "How to clean the nozzle"
print(f"Searching for: {test_query}")

# Convert question to numbers
query_vector = model.encode(test_query).tolist()

# Query Pinecone
try:
    results = index.query(
        vector=query_vector, 
        top_k=1, 
        include_metadata=True
    )

    if results['matches']:
        print("\n✅ Memory Search Success!")
        print(f"Top Match Found: {results['matches'][0]['metadata']['text'][:100]}...")
        print(f"Confidence Score: {results['matches'][0]['score']}")
    else:
        print("\n❌ Connected to Pinecone, but no data found in the index.")
except Exception as e:
    print(f"\n❌ Search Failed: {e}")