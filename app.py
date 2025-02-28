import os
import random
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import shutil

# Helper function to handle file uploads
def save_uploaded_files(uploaded_files, session_id):
    directory = os.path.join(os.getcwd(), session_id)
    os.makedirs(directory, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(directory, uploaded_file.name)
        if uploaded_file.type == "application/pdf":
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())
        else:
            bytes_data = uploaded_file.getvalue().decode("utf-8")
            with open(file_path, "w") as file:
                file.write(bytes_data)

# Function to create embeddings from uploaded documents
def create_embeddings(session_id):
    documents = []
    directory = os.path.join(os.getcwd(), session_id)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            documents += text_splitter.split_documents(docs)
        else:
            loader = TextLoader(file_path)
            text_documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            documents += text_splitter.split_documents(text_documents)
    return documents
st.set_page_config(page_title="Groq LLM SaaS Playground", page_icon="🤖", layout="wide")
# Add custom CSS to beautify the app
st.markdown("""
    <style>
        body {
            background-color: #f4f7fa;
            font-family: 'Helvetica', sans-serif;
        }
        .stButton button {
            background-color: #4CAF50; 
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stTextInput input {
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .stTextArea textarea {
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .stMarkdown {
            font-size: 1.2em;
            line-height: 1.5;
        }
        .chat-container {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .user-msg {
            background-color: #d1f7ff;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .ai-msg {
            background-color: #e8e8e8;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
        }
        .sidebar-subtitle {
            font-size: 14px;
            color: #888;
        }
        h1 {
            text-align: center;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Main application logic
def main():
    # Set page config must be the first Streamlit command
    

    # Header and attribution

    # Sidebar configuration
    st.sidebar.title("⚙️ Application Settings")
    st.sidebar.markdown("Interact with open-source LLMs using Groq API.")
    
    if 'groq_api_key' not in st.session_state:
        st.session_state['groq_api_key'] = ''
        st.session_state['authenticated'] = False
        uploaded_files = None

    # Handle authentication
    if not st.session_state['authenticated']:
        with st.expander("🔑 Enter Your GROQ API Key to Unlock", expanded=True):
            groq_api_key = st.text_input("GROQ API Key", type="password")
            uploaded_files = st.file_uploader("Please upload your document in .txt or .pdf format.", type=['pdf', 'txt'], accept_multiple_files=True)

            if st.button("Unlock Application"):
                if groq_api_key and uploaded_files:
                    st.session_state['groq_api_key'] = groq_api_key
                    st.session_state['authenticated'] = True
                    session_id = str(random.randint(1, 10000000000))
                    st.session_state['id'] = session_id
                    st.success("API Key authenticated successfully!")
                    save_uploaded_files(uploaded_files, st.session_state['id'])

                    with st.spinner('Embeddings are in process...'):
                        st.rerun()
                else:
                    st.error("Please enter a valid API key and upload files.")

        return

    # Once authenticated, proceed to chat and embeddings
    st.sidebar.subheader("Chat Settings")
    
    # Model selection dropdown
    model_name = st.sidebar.selectbox(
        'Choose a model',
        [ 'llama3-8b-8192','gemma-7b-it','gemma2-9b-it',
        'llama3-groq-70b-8192-tool-use-preview','llama3-groq-8b-8192-tool-use-preview',
        'distil-whisper-large-v3-en','llama-3.1-70b-versatile','llama-3.1-8b-instant','llama-3.2-11b-text-preview',
        'llama-3.2-11b-vision-preview','llama-3.2-1b-preview','llama-3.2-3b-preview','llama-3.2-90b-text-preview',
        'llama-3.2-90b-vision-preview','llama-guard-3-8b','llama3-70b-8192','whisper-large-v3','whisper-large-v3-turbo',
        'llava-v1.5-7b-4096-preview','mixtral-8x7b-32768']

    )

    # Initialize model and embedding
    model = ChatGroq(model_name=model_name, groq_api_key=st.session_state['groq_api_key'])
    parser = StrOutputParser()
    
    # Define the chat prompt template
    template = """
    Answer the question based on the context below. If you're unable to answer, 
    search online. If the answer is still unavailable, reply with "I don't know".
    
    Context: {context}
    
    Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # Create embeddings for the uploaded documents
    documents = create_embeddings(st.session_state['id'])
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

    # Set up the vector store for document retrieval
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)
    chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    # User interaction: Asking questions
    st.markdown("## 💬 Chat with the Model")
    user_question = st.text_area("📝 Ask a question:", placeholder="Type your query here...")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if user_question:
        with st.spinner("Thinking..."):
            response = chain.invoke(user_question)
            message = {"human": user_question, "AI": response}
            st.session_state['chat_history'].append(message)
            st.write("🤖 ChatBot:", response)

    # Display chat history with enhanced styling
    st.markdown("📜 Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['chat_history']:
            st.markdown(f'<div class="chat-container"><div class="user-msg"><strong>👤 You:</strong> {message["human"]}</div><div class="ai-msg"><strong>🤖 AI:</strong> {message["AI"]}</div></div>', unsafe_allow_html=True)

    # Sidebar buttons for clearing chat history and logging out
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.rerun()

    if st.sidebar.button("Log Out"):
        # Log out and clean up session
        st.session_state['authenticated'] = False
        st.session_state['groq_api_key'] = ''
        st.session_state['chat_history'] = []
        shutil.rmtree(os.path.join(os.getcwd(), st.session_state['id']))
        st.session_state['id'] = ''
        st.rerun()

if __name__ == "__main__":
    main()
