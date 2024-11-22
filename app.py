import os
from dotenv import load_dotenv
import random
import whisper
import tempfile
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import PyPDF2
import shutil


def helper(uploaded_files):
    os.mkdir(os.path.join(os.getcwd(),st.session_state['id']))
    for uploaded_file in uploaded_files:
        file_path = os.path.join(os.getcwd(),st.session_state['id'], uploaded_file.name)
        if(uploaded_file.type=="application/pdf"):
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())
        else:
            bytes_data = str(uploaded_file.read(),"utf-8")
            with open(file_path, "w") as file:
                file.write(bytes_data)

def main():
    st.set_page_config(page_title="Groq LLM SaaS Playground", page_icon="🤖", layout="wide")
    st.title("🤖 Groq LLM SaaS Playground")
    st.sidebar.title("⚙️ Application Settings")
    st.sidebar.markdown("Interact with open-source LLMs using Groq API.")
    
    if 'groq_api_key' not in st.session_state:
        st.session_state['groq_api_key'] = ''
        st.session_state['authenticated'] = False
        uploaded_files=""

    if not st.session_state['authenticated']:
        with st.expander("🔑 Enter Your GROQ API Key to Unlock", expanded=True):
            groq_api_key = st.text_input("GROQ API Key", type="password")
            uploaded_files = st.file_uploader("Choose a File",type=['pdf','txt'],accept_multiple_files=True)
            if st.button("Unlock Application"):
                if groq_api_key and uploaded_files:
                    st.session_state['groq_api_key'] = groq_api_key
                    st.session_state['authenticated'] = True
                    st.session_state['id']=str(random.randint(1,10000000000))
                    st.success("API Key authenticated successfully!")
                    helper(uploaded_files)
                    st.rerun()
                else:
                    st.error("Please enter a valid API key.")
        return

    st.sidebar.subheader("🤖 Chat Settings")

    # Model selection with more options
    m = st.sidebar.selectbox(
        'Choose a model',
        [ 'llama3-8b-8192','gemma-7b-it','gemma2-9b-it',
        'llama3-groq-70b-8192-tool-use-preview','llama3-groq-8b-8192-tool-use-preview',
        'distil-whisper-large-v3-en','llama-3.1-70b-versatile','llama-3.1-8b-instant','llama-3.2-11b-text-preview',
        'llama-3.2-11b-vision-preview','llama-3.2-1b-preview','llama-3.2-3b-preview','llama-3.2-90b-text-preview',
        'llama-3.2-90b-vision-preview','llama-guard-3-8b','llama3-70b-8192','whisper-large-v3','whisper-large-v3-turbo',
        'llava-v1.5-7b-4096-preview','mixtral-8x7b-32768']

    )
    model=ChatGroq(model_name=m,groq_api_key=st.session_state['groq_api_key'])
    

    parser = StrOutputParser()


    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    documents=[]
    directory = os.path.join(os.getcwd(),st.session_state['id'])
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if(filename[-3:]=="pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            documents=documents+ text_splitter.split_documents(docs)
        else:
            loader = TextLoader(file_path)
            text_documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            documents = documents+text_splitter.split_documents(text_documents)
    
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


    

    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    vectorstore2 = DocArrayInMemorySearch.from_documents(documents, embeddings)
    chain = (
        {"context": vectorstore2.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    
   
   
    st.markdown("## 💬 Chat with the Model")
    user_question = st.text_area("📝 Ask a question:", placeholder="Type your query here...")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    try:
        
        if user_question:
            with st.spinner("Thinking..."):
                response=chain.invoke(user_question)
                message = {"human": user_question, "AI": response}
                st.session_state['chat_history'].append(message)
                st.write("🤖 ChatBot:", response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    st.markdown("### 📜 Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['chat_history']:
            st.markdown(f"*👤 You*: {message['human']}", unsafe_allow_html=True)
            st.markdown(f"*🤖 AI*: {message['AI']}", unsafe_allow_html=True)

    # Sidebar Buttons
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.rerun()

    if st.sidebar.button("Log Out"):
        st.session_state['authenticated'] = False
        st.session_state['groq_api_key'] = ''
        st.session_state['chat_history'] = []
        shutil.rmtree(os.path.join(os.getcwd(),st.session_state['id']))
        st.session_state['id']=''
        st.rerun()

    



if __name__ == "__main__":
    main()