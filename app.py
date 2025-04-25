import streamlit as st
from knowledge_base import load_and_process_pdfs, chunk_documents, create_vector_store, get_relevant_context
from qa_bot import InsuranceAssistant

@st.cache_resource
def initialize_system():
    # Initialize all components
    raw_text = load_and_process_pdfs()
    chunks = chunk_documents(raw_text)
    vector_store = create_vector_store(chunks)
    bot = InsuranceAssistant()
    return vector_store, bot

def main():
    st.title("Insurance Policy Chatbot")
    vector_store, bot = initialize_system()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if prompt := st.chat_input("Ask about insurance policies:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # RAG pipeline
        try:
            context = get_relevant_context(prompt, vector_store)
            answer = bot.generate_response(prompt, context)
            
            # Fallback mechanism
            if not answer.strip() or len(answer.split()) < 5:
                answer = "Let me connect you to a human agent..."
        except Exception as e:
            answer = "Sorry, I'm having trouble. Please contact our support team."
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()