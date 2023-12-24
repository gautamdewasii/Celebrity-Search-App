# Integrate our code with OpenAI API
import os
# for secretly using our openai key
from constants import openai_api_key
# using openAI gpt llm model
from langchain.llms import OpenAI
# used for input prompt along with input variables
from langchain.prompts import PromptTemplate
# LLMChain to define prompt, llm, along with other parameters
from langchain.chains import LLMChain
# for using multiple input and output 
from langchain.chains import SequentialChain
# used to store our prompt and generated output in memory buffer.
from langchain.memory import ConversationBufferMemory
# for web app
import streamlit as st

# set the openai api key in os environment
os.environ["OPENAI_API_KEY"]=openai_api_key

# streamlit framwork :-
st.title("Celebrity search Application using Langchain")
input_text=st.text_input("Search any topic :- ")

# first prompt template :-
first_input_prompt=PromptTemplate(
    # name of input variable which will pass as an input
    input_variables=["name"],
    # prompt structure 
    template="Tell me about {name}"
    )

# defining memory for saving first prompt conversation
person_memory=ConversationBufferMemory(
    input_key='name',
    memory_key='chat_history'
)

# Using openai LLM
# temperature :- if 0 means agent did not generate creative result ( no risk),
# if temperature :- 1 means agent will generate creative result ( high risk) 
llm=OpenAI(temperature=0.7)
# defining chain structure for fist prompt
chain=LLMChain(
    # model for first prompt
    llm=llm,
    # name of the prompt
    prompt=first_input_prompt,
    # it will generate all steps in output terminal for better understanding 
    verbose=True,
    # defining result name
    output_key='person',
    # name of memory in which first prompt conversation will saved
    memory=person_memory)

# defining memory for saving second prompt conversation
dob_memory=ConversationBufferMemory(
    input_key='name',
    memory_key='chat_history'
)

# second prompt template :-
second_input_prompt=PromptTemplate(
    input_variables=["name"],
    template="In which year {name} was born. Mention only year?"
    )

chain2=LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key='dob',
    memory=dob_memory)

# defining memory for saving 3rd prompt conversation
desc_memory=ConversationBufferMemory(
    input_key='dob',
    memory_key='description_history'
)

# third prompt template :-
third_input_prompt=PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events happened around {dob} in the world?"
    )

chain3=LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    verbose=True,
    output_key='timelines',
    memory=desc_memory)


# this will combined all these three prompt and run them one by one
parent_chain=SequentialChain(
    # all chains ( prompt chains) 
    chains=[chain,chain2,chain3],
    # initial input variable name, can be multiple inputs
    input_variables=['name'],
    # all output ( result) after each chain execution
    output_variables=['person','dob','timelines'],
    # for better understanding 
    verbose=True
    )


if input_text:
    st.write(parent_chain({'name':input_text}))

    # for displaying saved memory data
    with st.expander('Person Name : '):
        st.info(person_memory.buffer)
    with st.expander('Date of Birth :- '):
        st.info(dob_memory.buffer)
    with st.expander('Major events : '):
        st.info(desc_memory.buffer)


