from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import WikipediaAPIWrapper
import pdfminer.high_level
import os 
#from apikey import apikey 
#from apikey import PCapikey
#from apikey import SPapikey
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import create_csv_agent, load_tools
from langchain.utilities import GoogleSerperAPIWrapper 
import openai
import pandas as pd

openai.api_key = st.secrets[apikey]

# Prompt the API with your desired text
#prompt_text = "give me a table with all the teams in the nba with their division?"
#response = openai.Completion.create(engine="text-davinci-003", prompt=prompt_text, max_tokens=600)

# Print the API's response
#print(response.choices[0].text)

# App framework

os.environ['OPENAI_API_KEY'] = st.secrets[apikey]

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(temperature=0, openai_api_key=st.secrets[apikey])
chain = load_qa_chain(llm, chain_type="stuff")


st.title('📊Financial Planning GPT- extreme beta :)')
prompt = st.text_input('Type in your prompt here, press enter') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='{topic}'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)


#print(chain.run(input_documents=docs, question=query))

if prompt: 

    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=4000)
    os.environ["SERPER_API_KEY"] = st.secrets[SPapikey]
    agent = GoogleSerperAPIWrapper()
   
    #print(search.run("What's a roth ira"))

    #From my own data
    from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import pdfminer.high_level
    import streamlit as st 
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SequentialChain 
    from langchain.memory import ConversationBufferMemory
    from langchain.utilities import WikipediaAPIWrapper 

    loader = UnstructuredPDFLoader("PDFs/BWiscool.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    #print (f'Now you have {len(texts)} documents')

    from langchain.vectorstores import Chroma, Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    import pinecone

    OPENAI_API_KEY = st.secrets[apikey]
    PINECONE_API_KEY = st.secrets[PCapikey]
    PINECONE_API_ENV = 'us-east-1-aws'

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)    
    # initialize pinecone
    pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console)
    )
    index_name = "test-index" # put in the name of your pinecone index here
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    query = "What is secure act 2.0?"
    docs = docsearch.similarity_search(query, include_metadata=True)


    from langchain.llms import OpenAI
    from langchain.chains.question_answering import load_qa_chain
    llm = OpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    query = prompt
    docs = docsearch.similarity_search(query, include_metadata=True)
    print(chain.run(input_documents=docs, question=query))


    result = "Brian's Data" + "\n\n" + chain.run(input_documents=docs, question=query) + "\n\n" +"OpenAI Result " + "\n" + response.choices[0].text + "\n\n" + "Google Result " + "\n\n" + agent.run(prompt)

    #Finalresponse = openai.Completion.create(
    #model="text-davinci-003",
    #prompt= "rephrase in 1000 words " + response.choices[0].text + agent.run(prompt), 
    #temperature=0.0,
    #ax_tokens=3000,
    #top_p=1.0,
    #frequency_penalty=0.0,
    #presence_penalty=0.0)

    #print(Finalresponse)
    #ans = Finalresponse.choices[0].text


    

    st.write(result)
    
    #if st.button('Click me!'):
     #   st.write('Button clicked!')
    
    #python -m streamlit run SimpleOpenAI-Streamlit.py
