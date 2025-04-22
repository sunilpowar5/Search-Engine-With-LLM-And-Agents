import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
# Fix for the deprecation warning - updated import path
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name='Search')

### streamlit sidebar for api key
st.sidebar.title('Settings')
api_key = st.sidebar.text_input('Enter your Groq API key:', type='password')

### Add environment variable setup for Streamlit Cloud
if not api_key:
    # Try to get from environment variables or secrets
    api_key = os.environ.get('GROQ_API_KEY', '')
    # If using Streamlit secrets (uncomment this):
    # api_key = st.secrets.get("GROQ_API_KEY", "")

if not api_key:
    st.error("Please provide a Groq API key in the sidebar or set it as an environment variable.")
    st.stop()

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': "Hi, I'm a ChatBot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input(placeholder='What is machine learning?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message('user').write(prompt)

    llm = ChatGroq(api_key=api_key, model='llama3-8b-8192', streaming=True)
    tools = [search, arxiv, wikipedia]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)
