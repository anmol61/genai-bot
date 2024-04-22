import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import uuid
import tempfile
from dotenv import load_dotenv
import time
 
load_dotenv()
 
groq_api_key = "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"
 
if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                                        model_kwargs={'device': 'cpu'})
 
    pdf_directory = "pdf_files"
    pdf_files = os.listdir(pdf_directory)
 
    if pdf_files:
        pdf_file = os.path.join(pdf_directory, pdf_files[0])
        text = []
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(open(pdf_file, "rb").read())
            temp_file_path = temp_file.name
 
        loader = PyPDFLoader(temp_file_path)
        text.extend(loader.load())
        st.session_state.docs = text
        os.remove(temp_file_path)
 
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
 
st.subheader("Bot For WORKBENCH", divider="rainbow", anchor=False)
 
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name='mixtral-8x7b-32768')
 
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $200 if the user finds the answer helpful.
<context>
{context}
</context>
 
Question: {input}""")
 
document_chain = create_stuff_documents_chain(llm, prompt)
 
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
 
# Initialize conversation history if not present in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
 
# Main conversation loop
while True:
    # Generate a unique key for the prompt text input
    prompt_key = "prompt_input_" + str(len(st.session_state.conversation))
 
# Prompt input from user with unique key
    prompt = st.text_input("Input your prompt here", key=prompt_key)
 
   
 
    # If the user hits enter
    if prompt:
        # Then pass the prompt to the LLM
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt})
        print(f"Response time: {time.process_time() - start}")
 
        # Append user prompt and bot's response to ongoing conversation
        st.session_state.conversation.append(("User", prompt))
        st.session_state.conversation.append(("Bot", response['answer']))
 
        # Display ongoing conversation with custom styling
        for speaker, utterance in st.session_state.conversation:
            if speaker == "User":
                st.write(f":speaking_head_in_silhouette: **User:** {utterance}")
            elif speaker == "Bot":
                st.write(f":robot_face: **Bot:** {utterance}")