import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain , LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


##Streamlit app
st.set_page_config(page_title="Text to Math problem solver ")
st.title("Text to math problem solver using GEMMA 2B model")
groq_api_key=st.sidebar.text_input(label="Groq API KEY", type="password")

if not groq_api_key:
    st.info("Please enter the GROQ API KEY to continue")
    st.stop()

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

#Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet and solving the math problem"
)

##Initialize the math problem
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="calculator",
    func=math_chain.run,
    description="A toole for solving math related problem"
)

prompt="""
You are a agent for solving mathematical problem. Logically arrive to the solution and converge to the 
solution and display it point wise for the question below
Question{question}
Answer:
"""
prompt=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

#combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt)

reasoning_tool=Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool used for solving logic-based and reasoning questions."
)

#initialize the agents
Assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbise=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"HI i am a math chatbot and i can answer math problems"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

#function for generating the response
def generate_response(question):
    response=Assistant_agent.invoke({'input':question})
    return response

question=st.text_area("Enter your question:","I have 5 banana and 7 apple if i eat 2 apple then how many apple remain with me")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=Assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)
    
    else:
        st.warning("please enter the question")