import streamlit as st
import json
from google.oauth2 import service_account
from google import genai
from pinecone import Pinecone

# Page Config
st.set_page_config(page_title="iLabs Smart Assistant", page_icon="🤖")
st.title("🤖 iLabs Smart Assistant")
st.caption("Expert guidance for 3D Printing and Makerspace technology.")

@st.cache_resource
def init_connections():
    try:
        # 1. Load Google Credentials
        creds_info = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        google_creds = service_account.Credentials.from_service_account_info(
            creds_info, 
            scopes=scopes
        )
        
        # 2. Initialize the Gemini Client with v1beta
        client = genai.Client(
            vertexai=True,
            project=st.secrets["PROJECT_ID"],
            location="asia-southeast1",
            credentials=google_creds,
            http_options={'api_version': 'v1beta'}  
        )
        
        # 3. Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])
        
        # We REMOVED SentenceTransformer here!
        return client, index
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None

# Initialize connections
client, index = init_connections()

if client:
    st.success("iLabs System Online!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history  
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about Ultimaker 3D printers or Laser Cutters..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. Generate Embedding using GOOGLE (768 dimensions)
            # This must match what you used in upload_to_pinecone.py
            embed_result = client.models.embed_content(
                model="text-embedding-004",
                contents=prompt
            )
            query_embedding = embed_result.embeddings[0].values

            # 2. Search Pinecone for context
            results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            context = "\n".join([res['metadata']['text'] for res in results['matches']])

            # 3. Build System Instruction
            instruction = f"""
            You are the Sunway iLabs Smart Assistant. 
            Use the following context from our technical manuals to answer the user:
            {context}
            
            If the answer isn't in the context, politely say you don't have that specific data.
            """

            # 4. Generate Content using Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt,
                config={'system_instruction': instruction}
            )

            full_response = response.text
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error generating response: {e}")