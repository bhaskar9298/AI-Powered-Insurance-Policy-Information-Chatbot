import streamlit as st
import os
from knowledge_base import load_and_process_pdfs, chunk_documents, create_vector_store, get_relevant_context
from qa_bot import InsuranceAssistant

# Set upload directory
UPLOAD_DIR = "insurance_pdfs/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("📄 PDF Upload & 🤖 Insurance Chatbot")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Track whether system should be reloaded
new_upload = False

# Save uploaded file
if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File saved to: {file_path}")
        # Set flag and upload time to refresh processing
        st.session_state.upload_time = str(os.path.getmtime(file_path))
        new_upload = True
    else:
        st.info("File already exists. No need to re-upload.")

# Use upload time to manage caching
upload_time = st.session_state.get("upload_time", "initial")

# System initialization, cached with dependency on upload_time
@st.cache_resource
def initialize_system(upload_time):
    raw_text = load_and_process_pdfs()  # Should read from `insurance_pdfs/`
    chunks = chunk_documents(raw_text)
    vector_store = create_vector_store(chunks)
    bot = InsuranceAssistant()
    return vector_store, bot

# Run chatbot only if at least one PDF exists
if len(os.listdir(UPLOAD_DIR)) == 0:
    st.warning("Please upload a PDF file to begin.")
else:
    vector_store, bot = initialize_system(upload_time)

    # Chatbot UI
    st.subheader("🤖 Insurance Policy Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about insurance policies:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            context = get_relevant_context(prompt, vector_store)
            answer = bot.generate_response(prompt, context)

            if not answer.strip() or len(answer.split()) < 5:
                answer = "Let me connect you to a human agent..."
        except Exception as e:
            answer = "Sorry, I'm having trouble. Please contact our support team."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
