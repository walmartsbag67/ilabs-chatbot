import streamlit as st

st.set_page_config(page_title="Sunway iLabs Dashboard", layout="wide")

st.title("Welcome to Sunway iLabs")
st.write("This is your main innovation dashboard. Use the floating button below to talk to our AI Assistant.")

# --- THE FLOATING BUTTON CODE ---
# Replace the URL with your actual deployed Streamlit link
chatbot_url = "https://ilabs-chatbot.streamlit.app" 

st.markdown(
    f"""
    <style>
    .floating-chat-icon {{
        position: fixed;
        bottom: 40px;
        right: 40px;
        width: 70px;
        height: 70px;
        background-color: #ed1c24; /* Sunway Red */
        color: white !important;
        border-radius: 50%;
        text-align: center;
        font-size: 35px;
        line-height: 70px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
        z-index: 9999;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.3s ease;
    }}
    .floating-chat-icon:hover {{
        background-color: #b31217;
        transform: scale(1.1);
    }}
    </style>

    <a href="{chatbot_url}" target="_blank" class="floating-chat-icon">
        💬
    </a>
    """,
    unsafe_allow_html=True
)