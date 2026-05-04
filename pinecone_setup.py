import pinecone
from pinecone import ServerlessSpec

pc = pinecone.Pinecone(api_key="pcsk_2fERb7_PUnzAn8cRe52YB5DbGMhL7wpzzRi8zzoW3H8dq7idGhVYza78fiBtAE7Uk3BTxD")

pc.create_index(
    name="printer-chatbot",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print("Index created!")