import streamlit as st
from config import settings
from src.qa_chain import ask_question
from utils.helpers import format_response, initialize_app_directories, log_message

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon=settings.APP_ICON,
    layout="wide"
)

# ------------------------------
# Initialize application directories and logging
# ------------------------------
initialize_app_directories()

# ------------------------------
# Title and description
# ------------------------------
st.title("📚 Research Paper Chat Assistant")
st.markdown("Ask questions about your documents. The assistant will answer based on your uploaded PDFs.")

# ------------------------------
# Initialize Streamlit session state flags if not present
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "clear_chat" not in st.session_state:
    st.session_state.clear_chat = False

# ------------------------------
# Clear chat history button
# ------------------------------
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.clear_chat = True  # set flag to prevent rendering old messages

# ------------------------------
# Display chat history only if chat is not cleared
# ------------------------------
if not st.session_state.clear_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    # Reset flag so future messages will display normally
    st.session_state.clear_chat = False

# ------------------------------
# Chat input box
# ------------------------------
if prompt := st.chat_input("Ask a question about your documents..."):
    if not prompt.strip():
        st.warning("Please enter a valid question.")
    else:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call ask_question with just the prompt
                    response = ask_question(prompt)

                    # Extract and format answer
                    answer = response.get("answer", "No answer found.")
                    formatted_response = format_response(answer)
                    st.markdown(formatted_response)

                    # Display sources if available
                    source_docs = response.get("source_documents", [])
                    if source_docs:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(source_docs):
                                st.markdown(
                                    f"**Chunk {i+1} from {source['source_file']}** "
                                    f"(Page: {source['page_number']})"
                                )
                                st.text(source['content_preview'])

                    # Add assistant response to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_response
                    })

                    # Logging
                    log_message(f"Q: {prompt}")
                    log_message(f"A: {answer}")

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    log_message(error_msg, level="error")



# ------------------------------
# Footer / Tip
# ------------------------------
st.markdown("---")
st.markdown(
    "💡 **Tip:** Ask specific questions about the content in your research papers for best results."
)
