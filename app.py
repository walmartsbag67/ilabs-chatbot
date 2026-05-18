import streamlit as st
import os
import base64
from google import genai
from google.genai import types
from google.oauth2 import service_account
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
        # 1. Map your online Streamlit Secrets directly into an authentication dictionary
        service_account_info = {
            "type": "service_account",
            "project_id": st.secrets["GCP_PROJECT_ID"],
            "private_key": st.secrets["GCP_PRIVATE_KEY"].replace(r'\n', '\n'), # Corrects the inline multi-line breaks
            "client_email": st.secrets["GCP_CLIENT_EMAIL"],
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        
        # 2. Parse the dictionary and explicitly add the Cloud Platform scope to prevent SDK authentication bugs
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info
        ).with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
        
        # 3. Initialize the Gemini Client routing explicitly via Vertex AI using your credentials
        client = genai.Client(
            vertexai=True,
            project=st.secrets["GCP_PROJECT_ID"],
            location=st.secrets["GCP_LOCATION"],
            credentials=credentials
        )
        
        # 4. Initialize Pinecone Vector Database
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])
        
        return client, index
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None, None

client, index = init_connections()
core_knowledge = load_core_knowledge()

# --- 3. UI STYLING & LAYOUT ---
# --- 🛠️ TEMPORARY KNOWLEDGE BASE DEBUGGER ---
with st.sidebar.expander("🔍 Debug Knowledge Sources"):
    st.write("**1. Direct local file check (model.md):**")
    if os.path.exists('model.md'):
        st.success("✅ model.md found successfully!")
        # Show the first 300 characters of your file to confirm it has text
        st.text_area("File Content Preview:", core_knowledge[:300], height=100)
    else:
        st.error("❌ model.md NOT found in root directory!")
        
    st.write("---")
    st.write("**2. Total database matches retrieved last turn:**")
    if 'combined_context' in locals() or 'combined_context' in globals():
        st.text_area("Pinecone Vector Context:", combined_context, height=150)
    else:
        st.info("No active query context loaded yet. Send a message to test.")
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
        st.image("Sunway-iLabs-Logo-AI-2025-837x1024 (1).png", width=80)

with col2:
    st.markdown("""
        <div style='margin-top: 10px;'>
            <h1 style='margin: 0;'>Sunway iLabs AI Assistant</h1>
            <p style='color: #808495;'>Grounded Knowledge System for Makerspace Labs</p>
        </div>
        """, unsafe_allow_html=True)

# --- 4. CHAT HISTORY INITIALIZATION & DISPLAY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Helper function to get query embeddings
def get_query_embedding(user_query: str) -> list:
    try:
        # Upgraded to text-embedding-004 for native, clean Vertex AI integration
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=user_query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        st.error(f"Detailed Embedding API Error: {e}")
        return None

# --- 5. STREAMLIT CHAT EXECUTION LOGIC ---
if prompt := st.chat_input("Ask about Sunway iLabs, 3D Printer, Laser Cutter."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching knowledge base..."):
        # Step A: Get embedding vector
        query_vector = get_query_embedding(prompt)
        
        if query_vector is None:
            st.error("Failed to process text embedding.")
        else:
            # Step B: Query Pinecone Vector Database
            combined_context = ""
            try:
                search_results = index.query(
                    vector=query_vector,
                    top_k=3,
                    include_metadata=True
                )
                
                # Extract text chunks from database matches safely checking alternate dictionary variants
                contexts = []
                for match in search_results.get('matches', []):
                    metadata = match.get('metadata', {})
                    # Checked standard key mappings to safeguard missing payloads
                    if 'text' in metadata:
                        contexts.append(metadata['text'])
                    elif 'content' in metadata:
                        contexts.append(metadata['content'])
                    elif 'context' in metadata:
                        contexts.append(metadata['context'])
                        
                combined_context = "\n\n".join(contexts)
                
            except Exception as e:
                st.error(f"Pinecone query failed: {str(e)}")

            # Step C: Build System Instruction & Run Inference via Gemini
            try:
                system_instruction = f"""
                You are the Sunway iLabs Smart Assistant.
                
                # STRICT OPERATING RULES:
                1. You are a CLOSED-KNOWLEDGE system.
                2. Use ONLY the information in the "LOCAL DATA" and "VECTOR CONTEXT" sections below.
                3. If the answer is NOT in the provided data, you MUST say: "I'm sorry, I don't have information on that specific topic in my current database."
                4. DO NOT mention "mandatory training" unless explicitly written in the data below.
                5. For any question about booking, provide only the URL and rules from the data below.
                
                # LOCAL DATA (model.md):
                {core_knowledge}
                
                # VECTOR CONTEXT (Pinecone Database):
                {combined_context}
                """

                # Generate Answer using Gemini 2.5 Flash
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        'system_instruction': system_instruction,
                        'temperature': 0.0,
                        'max_output_tokens': 1000
                    }
                )
                
                # Render and save assistant response
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"Error generating response: {e}")