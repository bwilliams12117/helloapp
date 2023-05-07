from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import WikipediaAPIWrapper
import pdfminer.high_level
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import create_csv_agent

import pandas as pd
import openai
openai.api_key = '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'

# Prompt the API with your desired text
prompt_text = "give me a table with all the teams in the nba with their division?"
response = openai.Completion.create(engine="text-davinci-003", prompt=prompt_text, max_tokens=600)

# Print the API's response
print(response.choices[0].text)

# App framework

OPENAI_API_KEY = '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'
os.environ['OPENAI_API_KEY'] = apikey

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")


st.title('📊Financial Planning GPT- extreme beta :)')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='{topic}'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=1) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)


#print(chain.run(input_documents=docs, question=query))

if prompt: 

    prompt_text = "give me a table with all the teams in the nba with their division?"
    response = openai.ChatCompletion.create(engine="text-davinci-003", prompt=prompt, max_tokens=600)


    st.write(response)

    #streamlit run SimpleOpenAI-Streamlit.py
