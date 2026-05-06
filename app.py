import streamlit as st
import os
import json
from google import genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Page Configuration
st.set_page_config(page_title="iLabs Smart Assistant", layout="wide")
st.title("🤖 iLabs Smart Assistant")
st.caption("Expert guidance for 3D Printing and Makerspace technology.")

# 2. Initialize Connections (Cached for performance)
@st.cache_resource
def init_connections():
    try:
        # Load Google Credentials from Streamlit Secrets
        creds_json = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        
        # Initialize the New Direct Gen AI Client (Bypasses Agent Deployment)
        client = genai.Client(
            vertexai=True,
            project=st.secrets["PROJECT_ID"],
            location="asia-southeast1", # Fast region for Malaysia
            credentials=creds_json
        )
        
        # Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])
        
        # Initialize Embedding Model for Search
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return client, index, embed_model
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None

client, index, embed_model = init_connections()

if client:
    st.success("iLabs System Online!")

# 3. Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handling User Input
if prompt := st.chat_input("Ask about Ultimaker 3D printers or Laser Cutters..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG: Search Pinecone for relevant iLabs documentation
    with st.spinner("Searching iLabs knowledge base..."):
        try:
            query_embedding = embed_model.encode(prompt).tolist()
            results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            
            # Combine search results into a context string
            context = "\n".join([res['metadata']['text'] for res in results['matches'] if 'text' in res['metadata']])
        except Exception as e:
            st.error(f"Pinecone Search Error: {e}")
            context = "No context available."

    # 5. Generate AI Response using Direct Vertex AI
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            # System Instruction to keep the AI focused
            instruction = f"""
            You are the iLabs Smart Assistant. 
            Use the following context to answer the user: {context}
            
            Rules:
            1. If the answer is not in the context, say you don't know but suggest contacting a mentor.
            2. Be professional and encouraging.
            3. Use a mix of English and Malay where appropriate.
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-flash", # Using the latest stable Flash model
                contents=prompt,
                config={'system_instruction': instruction}
            )
            
            final_text = response.text
            response_placeholder.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
            
        except Exception as e:
            st.error(f"Error generating response: {e}")