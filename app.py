from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain import hub
import os
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

# Get LLM
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.5,openai_api_key=os.getenv("OPENAI_API_KEY"))
print(llm)

# ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
"""
Answer the given question only based on the given context.
Please provide the most accurate response based on the question
<context>
{context}
<context>
                                 
Question: {input}
""")

chain = create_stuff_documents_chain(llm=llm,prompt=prompt)


st.title("Q&A Chatbot")

url = st.text_input("Enter the URL...")
url_ques = st.text_input("Enter your question...")

wiki_ques = st.text_input("Enter your question to search in wikipedia...")

if url and url_ques:
    if "url_db" not in st.session_state:
        st.session_state.loader = WebBaseLoader(web_path=url)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.url_docs = st.session_state.splitter.split_documents(documents=st.session_state.docs)
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.url_db = FAISS.from_documents(documents=st.session_state.url_docs,embedding=st.session_state.embeddings)

    retriver_url = st.session_state.url_db.as_retriever()

    # Combine the retriver and chain

    retrieval_chain_url = create_retrieval_chain(retriver_url,chain)
    response = retrieval_chain_url.invoke({'input':url_ques})
    st.write(response['answer'])

# Use of Agents
if wiki_ques:
    api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Can add many such tools
    tools = [wiki]


    agent_prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_openai_tools_agent(llm,tools,agent_prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

    agent_ans = agent_executor.invoke({'input':wiki_ques})
    st.write(agent_ans["output"])
    







