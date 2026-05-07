import streamlit as st
import json
from google.oauth2 import service_account
from google import genai
from pinecone import Pinecone

# --- Page Configuration ---
st.set_page_config(page_title="iLabs Smart Assistant", page_icon="Sunway-iLabs-Logo-AI-2025-837x1024 (1).png")
col1, col2 = st.columns([1, 4])
with col1:
    st.image("Sunway-iLabs-Logo-AI-2025-837x1024 (1).png", width=80)
with col2:
    st.markdown("""
        <h1 style='margin-bottom: 50px; font-size: 42px'>
            iLabs Smart Assistant
        </h1>
    """, unsafe_allow_html=True)

with st.sidebar:
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
        
# --- Initialize Connections ---
@st.cache_resource
def init_connections():
    try:
        # 1. Load Google Credentials from Streamlit Secrets
        creds_info = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        
        # Define Scopes for Vertex AI
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        google_creds = service_account.Credentials.from_service_account_info(
            creds_info, 
            scopes=scopes
        )
        
        # 2. Initialize Gemini 2.5 Flash Client
        client = genai.Client(
            vertexai=True,
            project=st.secrets["PROJECT_ID"],
            location="asia-southeast1",
            credentials=google_creds,
        )
        
        # 3. Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])
        
        return client, index
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None

client, index = init_connections()

if client:
    st.success("iLabs System Online!")



# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input & Logic ---
if prompt := st.chat_input("Ask about Sunway iLabs, Ultimaker 3D printers or Laser Cutters"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Generate Embedding
            embed_result = client.models.embed_content(
                model="text-embedding-004", # Double check this name in GC Console
                contents=prompt
            )
            query_embedding = embed_result.embeddings[0].values

            # 2. Search Pinecone
            search_results = index.query(
                vector=query_embedding, 
                top_k=3, 
                include_metadata=True
            )
            
            context = "\n---\n".join([res['metadata']['text'] for res in search_results['matches']])

            # 3. Build Instruction
            system_instruction = f"You are the iLabs Assistant. Use this context: {context}"

            # 4. Generate Response (Updated model name to stable version)
            response = client.models.generate_content(
                model="gemini-2.5-flash", # Changed from 2.5 to 1.5 for stability
                contents=prompt,
                config={'system_instruction': system_instruction}
            )

            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            st.error(f"Error: {e}")