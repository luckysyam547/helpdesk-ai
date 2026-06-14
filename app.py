import os
import streamlit as st
import torch
from inference import load_model, generate_response

st.set_page_config(page_title="HelpDesk AI", page_icon="💬", layout="centered")

st.markdown("""
<style>
.chat-container {
    background-color: #e5ddd5;
    border-radius: 10px;
    padding: 15px;
    height: 450px;
    overflow-y: auto;
}
.bubble-user {
    background-color: #dcf8c6;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    max-width: 70%;
    margin-left: auto;
    text-align: right;
}
.bubble-bot {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    max-width: 70%;
    box-shadow: 0px 1px 1px rgba(0,0,0,0.1);
}
.header-bar {
    background-color: #075e54;
    color: white;
    padding: 12px;
    border-radius: 10px 10px 0 0;
    font-weight: bold;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-bar">💬 HelpDesk AI</div>', unsafe_allow_html=True)

@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model("chatbot_model.pt", device)

if not os.path.exists("chatbot_model.pt"):
    st.error("Model not found. Please run python train.py first.")
    st.stop()

encoder, decoder, word2index, index2word, max_length, device = get_model()

if "messages" not in st.session_state:
    st.session_state.messages = [("bot", "Hello! How can I help you today?")]

chat_html = '<div class="chat-container">'
for sender, msg in st.session_state.messages:
    cls = "bubble-user" if sender == "user" else "bubble-bot"
    chat_html += f'<div class="{cls}">{msg}</div>'
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type a message...", label_visibility="collapsed",
                                placeholder="Type a message...")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append(("user", user_input))
    response = generate_response(
        user_input, encoder, decoder, word2index, index2word, max_length, device)
    st.session_state.messages.append(("bot", response))
    st.rerun()

st.caption("Powered by PyTorch Seq2Seq with Luong Attention.")
