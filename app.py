import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Load API keys from secrets or environment
openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Set Streamlit page config
st.set_page_config(page_title="AI Course Chatbot", layout="wide")

# --- Sidebar ---
st.sidebar.title("📄 Welcome to Generative AI Course")
st.sidebar.image("Generative-AI.jpg", use_container_width=True)  # Image below title
st.sidebar.info("Register below")
st.sidebar.info("https://docs.google.com/forms/d/e/1FAIpQLSd4ZFuvYLqSvPsALpO_czpukxKxEWN45L43SHWuFwAFRv2ZOg/viewform")



st.subheader("💬 Chat with the Course Assistant")

# Load embedding model and vectorstore
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# Set up LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.3)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Custom CSS to ensure the input box stays fixed at the bottom
st.markdown(
    """
    <style>
    .block-container {
        display: flex;
        flex-direction: column;
        height: 80vh; /* Ensure the layout is tall enough */
        justify-content: flex-start; /* Keep the elements at the top */
    }
    .stTextInput > div > input {
        position: sticky; /* Fix the input box at the top */
        top: 90%; /* Adjust the input box position */
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# User input
query = st.text_input("You:", "", key="input", label_visibility="collapsed")

# Handle query
if query:
    # Display user query (left side)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"🧑‍💻 **You:** {query}")

    # Generate response from AI
    response = qa_chain.run(query)

    # Display AI response (right side)
    with col2:
        st.markdown(f"🤖 **AI:** {response}")
