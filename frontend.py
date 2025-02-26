from langchain.indexes import VectorstoreIndexCreator
from langchain_core.documents import Document
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import streamlit as st 
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI


st.title('Advertisement Recomendation System')
prompt2=st.text_input('enter website here') 
prompt3 = st.text_input('Plug in your question here')

prompt4 = st.text_input('Produce an image for above?') 

new_db1 = FAISS.load_local(r"C:\\Users\\jasvi\\OneDrive\\Desktop\\sem6\\project_mcd\\chain\\pdf_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever_pdf = new_db1.as_retriever()

api_wrapper=  WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

#a=input("enter the website address:")
retriever = None
if prompt2:
    loader=WebBaseLoader(prompt2)
    docs = loader.load()
    documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)

    vectordb=FAISS.from_documents(documents,OpenAIEmbeddings())

    retriever=vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
retriever_tool=create_retriever_tool(retriever, "product_ad","Search for information about the product for any questions asked")
retriever_pdf_tool=create_retriever_tool(retriever, "pdf_search","For any questions about marketing strategies, advertisement startegies refer this")
    
tools=[wiki,retriever_tool, retriever_pdf_tool]
# os.environ["OPENAI_API_KEY"] = "YOUR OPENAI SECRET KEY"

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")


agent=create_openai_tools_agent(llm,tools,prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)


#b=input("ask your question:")
prompt_memory= ConversationBufferMemory(return_messages=True)  
prompti=None
ans=None

def answer_it(prompt3):
    answer=agent_executor.invoke({"input":prompt3})
    prompt_memory.save_context( {"input":str(prompt3)}, {"output":str(answer)})
    return answer

if prompt3:
    answer=answer_it(prompt3)
    ans=answer
    st.write(answer) 
    prompti=st.text_input("any further questions?") 
    answer1=answer_it(prompti)
    st.write(answer1)


    
if prompt4:
    from openai import OpenAI
    client = OpenAI()
    
    response = client.images.generate(
    model="dall-e-3",
    prompt=str(ans),
    size="1024x1024",
    quality="standard",
    n=1,
    )
    image_url = response.data[0].url
    st.image(image_url)




st.button("Reset", type="primary")
if st.button("Show History"):
    conversation_history = prompt_memory.load_memory_variables({})
    st.write("Conversation History:")
    '''for prompt, answer in conversation_history:
        st.write(f"User: {prompt}")
        st.write(f"AI: {answer}")'''
    st.write(conversation_history)
else:
    st.write("No conversation history available.")
