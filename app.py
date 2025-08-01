import streamlit as st
from qa_engine import get_answer

st.set_page_config(page_title="StudyMate: AI PDF Q&A", layout="wide")
st.title("📚 StudyMate: AI-Powered PDF Q&A")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

st.markdown("---")

# Display chat history
for msg in st.session_state.conversation:
    st.markdown(f"**You:** {msg['user']}")
    st.markdown(f"**StudyMate:** {msg['bot']}")

# Input and send
user_input = st.text_input("Ask a question about your PDF:", key="user_input")

if st.button("Send") and user_input:
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        with st.spinner("Generating answer..."):
            answer = get_answer(user_input, uploaded_file)
            st.session_state.conversation.append({"user": user_input, "bot": answer})
            st.rerun()
