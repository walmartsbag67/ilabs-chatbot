import streamlit as st
from streamlit_float import *
from google import genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os



# --- 1. CONFIGURATION & STYLING ---
# Replaced Rocket with official iLabs Logo
st.set_page_config(
    page_title="iLabs Smart Assistant", 
    layout="wide", 
    page_icon="Sunway-iLabs-Logo-AI-2025-837x1024 (1).png"
)
float_init()

# Custom CSS for the Chat Interface & Animation
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 5px; }
    
    /* The Animated Floating Icon CSS */
    #ilabs-animated-widget {
        width: 70px; height: 70px;
        background: linear-gradient(135deg, #87CEEB, #00BFFF);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 15px rgba(0, 191, 255, 0.5);
        cursor: pointer; overflow: hidden;
    }
    .icon { position: absolute; width: 40px; height: 40px; transition: opacity 1s ease-in-out; filter: brightness(0) invert(1); }
    @keyframes printerFade { 0%, 45% { opacity: 1; } 55%, 100% { opacity: 0; } }
    @keyframes laserFade { 0%, 45% { opacity: 0; } 55%, 100% { opacity: 1; } }
    .icon-1 { animation: printerFade 5s infinite; }
    .icon-2 { animation: laserFade 5s infinite; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALIZE SERVICES ---
@st.cache_resource
def init_connections():
    # Gemini - Fixed for Streamlit Secrets
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Pinecone - Fixed for Streamlit Secrets
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("printer-chatbot") 
    
    # Embedding Model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return client, index, embed_model

client, index, embed_model = init_connections()

# Load model.md
def load_core_knowledge():
    if os.path.exists('model.md'):
        with open('model.md', 'r', encoding='utf-8') as f:
            return f.read()
    return "Standard Sunway iLabs safety procedures."

core_knowledge = load_core_knowledge()


# --- 3. UI LAYOUT ---
# 1. Ensure the overall page padding is reduced
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# 2. Create columns
col1, col2 = st.columns([0.15, 0.85])

with col1:
    # Wrap the image in a div and use a negative margin to pull it UP
    st.markdown("""
        <div style='margin-top: -300px;'>
    """, unsafe_allow_html=True)
    
    st.image("Sunway-iLabs-Logo-AI-2025-837x1024 (1).png", width=85)
    
    st.markdown("</div>", unsafe_allow_html=True)
    

with col2:
    # This moves the text to the right and aligns it vertically
    st.markdown("""
        <div style='margin-top: 0px;'>
            <h1 style='margin: 0; line-height: 1;'>Sunway iLabs AI Assistant</h1>
            <p style='margin: 0; color: #808495;'>Expert help for 3D Printing and Laser Cutting</p>
        </div>
    """, unsafe_allow_html=True)


# Sidebar for Lab Status
with st.sidebar:
    # This ensures the iLabs logo is prominent at the top of the sidebar
    st.image("Sunway-iLabs-Logo-AI-2025-837x1024 (1).png", width=100)
    st.header("Lab Status")
    st.success("Ultimaker 3: ONLINE")
    st.success("Laser Cutter 5030: ONLINE")

# --- 4. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about the 3D printers or Laser cutter..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG: Search Pinecone
    query_vector = embed_model.encode(prompt).tolist()
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    manual_context = results['matches'][0]['metadata']['text'] if results['matches'] else ""

    # Build AI Prompt
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
    full_prompt = f"""
    ROLE: Sunway iLabs Assistant.
    CORE RULES: {core_knowledge}
    MANUAL INFO: {manual_context}
    HISTORY: {history_str}
    QUESTION: {prompt}
    """

    with st.chat_message("assistant"):
        response = client.models.generate_content(model="gemini-3-flash", contents=full_prompt)
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

# --- 5. THE ANIMATED FLOATING WIDGET ---
container = st.container()
with container:
 import base64



