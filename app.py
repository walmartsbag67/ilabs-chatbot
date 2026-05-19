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
st.markdown("""
    <style>
    /* Add !important right before the semicolon to force the color override */
    .main { background-color: #1a1c23 !important; } 
    .stChatMessage { border-radius: 15px; }
    .block-container { padding-top: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

# Full-aspect resolution loader
logo_file = "Sunway-iLabs-Logo-AI-2025-837x1024 (1).png"

if os.path.exists(logo_file):
    # Added padding and a bounding-box restriction to prevent edge cropping
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 25px; padding: 20px 0;">
            <img src="data:image/png;base64,{get_base64(logo_file)}" 
                 style="height: 80px; width: auto; object-fit: contain; padding: 2px; max-width: 100%;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; line-height: 1.2;">Sunway iLabs Chatbot</h1>
                <p style="margin: 5px 0 0 0; color: #808495; font-size: 1rem;">Ask about Sunway iLabs, 3D Printer, Laser Cutter.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style='margin-top: 10px;'>
            <h1 style='margin: 0;'>Sunway iLabs Chatbot</h1>
            <p style='color: #808495;'>Ask about Sunway iLabs, 3D Printer, Laser Cutter.</p>
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
if prompt := st.chat_input("..."):
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