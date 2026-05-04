import os
from google import genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---
PROJECT_ID = "project-67f295a8-2e8f-45e2-a81"
PINECONE_API_KEY = "pcsk_2fERb7_PUnzAn8cRe52YB5DbGMhL7wpzzRi8zzoW3H8dq7idGhVYza78fiBtAE7Uk3BTxD" 
INDEX_NAME = "printer-chatbot" 

# --- 2. INITIALIZATION ---
# Initialize Gemini
client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize Embedding Model (for searching manuals)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Memory
chat_history = [] 

# --- 3. LOAD CORE KNOWLEDGE (model.md) ---
def load_core_knowledge():
    try:
        with open('model.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Standard Sunway iLabs 3D printing and safety procedures."

core_knowledge = load_core_knowledge()

# --- 4. CHAT FUNCTION ---
def ask_chatbot(question):
    global chat_history

    # A. Search Pinecone for specific manual snippets
    query_vector = embed_model.encode(question).tolist()
    results = index.query(vector=query_vector, top_k=2, include_metadata=True)
    
    manual_context = ""
    if results['matches']:
        manual_context = results['matches'][0]['metadata']['text']
    else:
        manual_context = "No specific manual snippet found for this query."

    # B. Format History (Last 3 turns)
    history_string = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history[-3:]])

    # C. Build the Augmented Prompt
    prompt = f"""
    ROLE: You are the Sunway iLabs 3D Printing Assistant. 
    
    CORE GUIDELINES (from model.md):
    {core_knowledge}

    SPECIFIC MANUAL CONTEXT:
    {manual_context}

    CONVERSATION HISTORY:
    {history_string}
    
    GUARDRAILS & INSTRUCTIONS:
    1. Only answer questions about 3D printing, laser cutting, or Sunway iLabs equipment.
    2. If the user asks about unrelated topics (food, sports, other homework), say: 
       "I am here to help with makerspace equipment only. Please stay on topic!"
    3. Use the CORE GUIDELINES and MANUAL CONTEXT to provide accurate, safe advice.
    4. If you don't know the answer, tell the student to contact a Lab Supervisor.

    STUDENT'S QUESTION: {question}
    """

    # D. Generate Response
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # E. Save to Memory
        chat_history.append((question, response.text))
        return response.text

    except Exception as e:
        return f"System Error: {str(e)}"

# --- 5. TERMINAL INTERFACE ---
print("\n" + "="*50)
print("🚀 SUNWAY iLABS SMART ASSISTANT IS ONLINE")
print("Knowledge Source: model.md + Pinecone")
print("Status: Memory Active | Guardrails Enabled")
print("="*50)

while True:
    user_input = input("\nStudent: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("\nAI Assistant: Happy making! Closing session...")
        break
    
    if not user_input:
        continue

    answer = ask_chatbot(user_input)
    print(f"\nAI Assistant: {answer}")