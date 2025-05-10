import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API key from secrets or environment variable
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Set Streamlit page config
st.set_page_config(page_title="AI Course Chatbot", layout="wide")

# --- Sidebar ---
st.sidebar.title("📄 Welcome to Generative AI Course")
st.sidebar.image("Generative-AI.jpg", use_container_width=True)  # Image below title
st.sidebar.info("Register below")
st.sidebar.info("https://docs.google.com/forms/d/e/1FAIpQLSd4ZFuvYLqSvPsALpO_czpukxKxEWN45L43SHWuFwAFRv2ZOg/viewform")

# --- Main Section ---
st.subheader("💬 Chat with the Course Assistant")

# Load embedding model and vectorstore
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# Set up LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.3)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Input field (positioned right under the title)
query = st.text_input("Type your question here:", "", key="input")

# Handle user query
if query:
    # Display user query (left)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("🧑‍💻 **You:**")
        st.info(query)

    # Get AI response
    response = qa_chain.run(query)

    # Display AI response (right)
    with col2:
        st.markdown("🤖 **AI:**")
        st.success(response)
