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
            # 1. Search Logic
            embed_result = client.models.embed_content(
                model="text-embedding-004",
                contents=prompt
            )
            query_vector = embed_result.embeddings[0].values
            results = index.query(vector=query_vector, top_k=1, include_metadata=True)
            manual_context = results['matches'][0]['metadata']['text'] if results['matches'] else ""

            core_knowledge = load_core_knowledge()

            # 2. The "Unbreakable" System Instruction
            # We put the URL at the VERY BOTTOM so the AI sees it last
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

# ADDITIONAL CONTEXT:
{manual_context}
"""

# 3. Apply the "Strictness" settings
response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=prompt,
    config={
        'system_instruction': system_instruction,
        'temperature': 0.0,      # CRITICAL: 0.0 makes it a literal robot
        'top_p': 0.1,            # Limits word variety
        'max_output_tokens': 250 # Keeps it from rambling
    }
)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            st.error(f"Error: {e}")