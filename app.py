import streamlit as st
from google import genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

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
    # Initialize Gemini Client using Secrets label
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Initialize Pinecone using Secrets label
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("printer-chatbot")
    
    # Load Embedding Model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return client, index, embed_model

client, index, embed_model = init_connections()

# --- 3. HELPER FUNCTIONS ---
def get_context(query):
    # Convert user query to vector
    query_em = embed_model.encode(query).tolist()
    
    # Search Pinecone for relevant 3D printer data
    results = index.query(vector=query_em, top_k=3, include_metadata=True)
    
    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n"
    return context

# --- 4. CHAT INTERFACE ---
st.title("🤖 iLabs Smart Assistant")
st.caption("Expert guidance for 3D Printing and Makerspace technology.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about the Ultimaker 3 or Laser Cutter..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Logic: Get context from Pinecone
    context = get_context(prompt)
    
    full_prompt = f"""
    You are the iLabs Smart Assistant. Use the following context to answer the user's question accurately.
    Context: {context}
    User Question: {prompt}
    """

    # Generate AI Response
    with st.chat_message("assistant"):
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=full_prompt
            )
            assistant_response = response.text
            st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            st.error(f"An error occurred: {e}")