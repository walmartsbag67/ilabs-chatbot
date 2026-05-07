import streamlit as st
import os
import base64
from google import genai
from pinecone import Pinecone

# --- 1. CORE LOGIC & FUNCTION DEFINITIONS ---
# (Must be at the top to prevent "NameError")

def load_core_knowledge():
    """Reads the local model.md file to use as the primary knowledge source."""
    if os.path.exists('model.md'):
        with open('model.md', 'r', encoding='utf-8') as f:
            return f.read()
    return "Standard Sunway iLabs safety procedures."

def get_base64(file_path):
    """Converts local image to base64 for the UI."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# --- 2. INITIALIZE SERVICES ---
@st.cache_resource
def init_connections():
    try:
        # Gemini Client with v1beta versioning for stability in 2026
        client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"],
            http_options={'api_version': 'v1beta'}
        )
        # Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])
        return client, index
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None, None

# Run initialization
client, index = init_connections()
core_knowledge = load_core_knowledge()

# --- 3. UI STYLING & LAYOUT ---
st.set_page_config(
    page_title="iLabs Smart Assistant", 
    layout="wide", 
    page_icon="Sunway-iLabs-Logo-AI-2025-837x1024 (1).png"
)

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; }
    .block-container { padding-top: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([0.15, 0.85])
with col1:
    if os.path.exists("Sunway-iLabs-Logo-AI-2025-837x1024 (1).png"):
        st.image("Sunway-iLabs-Logo-AI-2025-837x1024 (1).png", width=85)

with col2:
    st.markdown("""
        <div style='margin-top: 10px;'>
            <h1 style='margin: 0;'>Sunway iLabs AI Assistant</h1>
            <p style='color: #808495;'>Grounded Knowledge System for Makerspace Labs</p>
        </div>
        """, unsafe_allow_html=True)

# --- 4. CHAT INTERFACE & LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about equipment or bookings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. RAG Search (Retrieval)
            embed_result = client.models.embed_content(
                model="text-embedding-004",
                contents=prompt
            )
            query_vector = embed_result.embeddings[0].values
            search_results = index.query(vector=query_vector, top_k=1, include_metadata=True)
            manual_context = search_results['matches'][0]['metadata']['text'] if search_results['matches'] else ""

            # 2. STRICT SYSTEM INSTRUCTION (The Knowledge Wall)
            system_instruction = f"""
You are the Sunway iLabs Smart Assistant.

# STRICT RULES:
1. You are a CLOSED-KNOWLEDGE system. 
2. Use ONLY the information provided in the "LOCAL DATA" section below.
3. If a user asks a question that is NOT covered in the LOCAL DATA, you must say: "I'm sorry, I don't have information on that specific topic in my current database."
4. DO NOT use your own internal knowledge to answer questions. 
5. DO NOT mention "mandatory training" or "certification" unless it is in the LOCAL DATA.

# LOCAL DATA FROM model.md:
{core_knowledge}

# ADDITIONAL CONTEXT FROM MANUALS:
{manual_context}
"""

            # 3. Generate Content
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={
                    'system_instruction': system_instruction,
                    'temperature': 0.0,
                    'max_output_tokens': 200
                }
            )
            
            full_response = response.text
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error generating response: {e}")