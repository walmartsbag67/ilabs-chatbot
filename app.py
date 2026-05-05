import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import json

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="iLabs Smart Assistant", page_icon="🤖")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALIZE SERVICES ---
@st.cache_resource
def init_connections():
    # 1. Handle Google Credentials from Streamlit Secrets
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
    creds_dict = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

    if "private_key" in creds_dict:
        # This fixes the PEM formatting error by converting escaped newlines
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        
        # Save the cleaned dictionary to a temporary file
        with open("google_creds.json", "w") as f:
            json.dump(creds_dict, f)
    
        # Tell Google Cloud where to find the file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_creds.json"    

    else:
        st.error("Google Credentials not found in Secrets! Please check your Streamlit Cloud settings.")
        st.stop()

    # 2. Initialize Vertex AI
    # Automatically pulls the project_id from your credentials dictionary
    vertexai.init(project=creds_dict["project_id"], location="us-central1")
    model = GenerativeModel("gemini-1.5-flash") 
    
    # 3. Initialize Pinecone
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("printer-chatbot")
    
    # 4. Load Embedding Model
    # Note: Ensure 'sentence-transformers' is in your requirements.txt
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return model, index, embed_model

# Call the function to set everything up
model, index, embed_model = init_connections()

# --- 3. HELPER FUNCTIONS ---
def get_context(query):
    # Convert the user query into a vector (embedding)
    query_em = embed_model.encode(query).tolist()
    
    # Search Pinecone for the most relevant data matches
    results = index.query(vector=query_em, top_k=3, include_metadata=True)
    
    # Combine the matching texts into a single context string
    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n"
    return context

# --- 4. CHAT INTERFACE ---
st.title("🤖 iLabs Smart Assistant")
st.caption("Expert guidance for 3D Printing and Makerspace technology.")

# Initialize chat history session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Box
if prompt := st.chat_input("Ask anything"):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG (Retrieval-Augmented Generation) Logic
    # 1. Retrieve relevant technical context from Pinecone
    context = get_context(prompt)
    
    # 2. Build the augmented prompt
    full_prompt = f"""
    You are the iLabs Smart Assistant. Use the following context to answer the user's question accurately.
    If the answer isn't in the context, use your general knowledge but mention you are doing so.
    
    Context: {context}
    User Question: {prompt}
    """

    # 3. Generate AI Response using Vertex AI
    with st.chat_message("assistant"):
        try:
            response = model.generate_content(full_prompt)
            assistant_response = response.text
            st.markdown(assistant_response)
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            st.error(f"An error occurred during response generation: {e}")