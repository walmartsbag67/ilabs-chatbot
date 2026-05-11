import streamlit as st
import os
import base64
from google import genai
from pinecone import Pinecone

# --- 1. CONFIGURATION & LOGIC LOADERS ---
st.set_page_config(
    page_title="iLabs Smart Assistant", 
    layout="wide", 
    page_icon="Sunway-iLabs-Logo-AI-2025-837x1024 (1).png"
)

def load_core_knowledge():
    """Reads the local model.md file to use as the primary knowledge source."""
    if os.path.exists('model.md'):
        with open('model.md', 'r', encoding='utf-8') as f:
            return f.read()
    return "No local knowledge found. Please check model.md."

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
        # Gemini Client with v1beta versioning
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

client, index = init_connections()
core_knowledge = load_core_knowledge()

# --- 3. UI STYLING & LAYOUT ---
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

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about equipment or bookings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. RAG Search (768-dim)
            embed_result = client.models.embed_content(
                model="models/text-embedding-004",
                contents=prompt
            )
            query_vector = embed_result.embeddings[0].values
            search_results = index.query(vector=query_vector, top_k=1, include_metadata=True)
            manual_context = search_results['matches'][0]['metadata']['text'] if search_results['matches'] else ""

            # 2. THE STRICT SYSTEM INSTRUCTION
            system_instruction = f"""
            You are the Sunway iLabs Smart Assistant.
            
            # STRICT OPERATING RULES:
            1. You are a CLOSED-KNOWLEDGE system.
            2. Use ONLY the information in the "LOCAL DATA" section below.
            3. If the answer is NOT in the LOCAL DATA, you MUST say: "I'm sorry, I don't have information on that specific topic in my current database."
            4. DO NOT mention "mandatory training", "certification", or "file preparation" unless explicitly written in LOCAL DATA.
            5. For any question about booking, provide only the URL and rules from LOCAL DATA.
            
            # LOCAL DATA (model.md):
            {core_knowledge}
            
            # ADDITIONAL CONTEXT:
            {manual_context}
            """

            # 3. Generate Content with Literal Constraints
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    'system_instruction': system_instruction,
                    'temperature': 0.0,
                    'max_output_tokens': 250
                }
            )
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            st.error(f"Error generating response: {e}")