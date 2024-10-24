from typing import List, Union, Annotated
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from typing import Sequence, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import getpass
import os
import functools
import operator
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# API 키 설정
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY
os.environ["TAVILY_API_KEY"]=TAVILY_API_KEY

#랭스미스 키&프로젝트 입력
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "1021"
os.environ["LANGCHAIN_API_KEY"] = "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" #개인API

#Tools
tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()



# Streamlit 앱 설정
st.title("HSAD 챗봇 💬")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 대화 내용을 저장할 리스트 초기화


# 상수 정의
class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """

    USER = "user"  # 사용자 메시지 역할
    ASSISTANT = "assistant"  # 어시스턴트 메시지 역할


class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = "text"  # 텍스트 메시지
    FIGURE = "figure"  # 그림 메시지
    CODE = "code"  # 코드 메시지
    DATAFRAME = "dataframe"  # 데이터프레임 메시지


# 메시지 관련 함수
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)  # 텍스트 메시지 출력
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)  # 그림 메시지 출력
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=False):
                            st.code(
                                message_content, language="python"
                            )  # 코드 메시지 출력
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)  # 데이터프레임 메시지 출력
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")
########################################################################################################


#supervisor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal

members = ["Researcher", "Visualizer", "Planner"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH. Carefully consider if visualization is necessary"
    " before assigning work to the Visualizer. Only assign visualization tasks"
    " when explicitly requested or when it significantly aids in understanding"
    " complex data or results."
    " your first job is to call Planner node to get specified questions."
    " Once the Planner node returns end or says it is time to conduct research, start research"
)

options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal['FINISH', 'Researcher', 'Visualizer', 'Planner']

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4o")

def supervisor_agent(state):
    supervisor_chain = (
        prompt
        | llm.with_structured_output(routeResponse)
    )
    return supervisor_chain.invoke(state)

#researcher, Visualizer(ex-coder), reviewer, Planner(ex-feedback) 에이전트, 노드 정의

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

#code_agent 정의
code_agent_prompt= (
    "You are a Visualizer tasked with executing code and visualizing results."
    " Make sure the pictures and letters are arranged properly so that they can be seen clearly."
    " Don't skip the text sources you need to visualize."
    " Once the visualization is complete, return 'END' and ignore additional visualization requests."
    " If there is an error in visualization, do not print the result on the screen."
)
code_agent = create_react_agent(llm, tools=[python_repl_tool],state_modifier=code_agent_prompt)
code_node = functools.partial(agent_node, agent=code_agent, name="Visualizer")

#research_agent
def create_agent(llm, tools):
    """Create an agent."""
    research_agent_prompt= (
    "You are a helpful Research AI assistant, collaborating with other assistants."
    " Use the provided tools to progress towards answering the question."
    " Incorporate user input from the previous step where relevant."
    " Generate queries based on the message that Planner node have confirmed with user."
    " If you have received feedback from the reviewer that the web search results corresponding to the query you generated are appropriate, generate the research results with 'FINAL ANSWER'."
    " Also, when Reviewer says it is okay to generate the 'FINAL ANSWER' based on the provided information, return 'FINAL ANSWER' with your result to close your research."
    " If you have tried searching several times but the Reviewer asks you to create FINAL ANSWER based on the results so far due to a lack of search information, increase the reliability by stating that the search results are insufficient in FINAL ANSWER.")
    prompt = ChatPromptTemplate.from_messages([("system", research_agent_prompt), MessagesPlaceholder(variable_name="messages")])
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

research_agent = create_agent(llm, [tavily_tool])

# research_agent_node
def research_agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result]
    }

research_node = functools.partial(research_agent_node, agent=research_agent, name="Researcher")

# Reviewer agent
system_prompt = (
    "You are a reviewer AI that checks if the web search results match the query."
    " You will be provided with a query and the search results."
    " Compare the web search results with the query and determine if they are relevant with your reason."
    " if not, suggest the researcher to search again trying other specified or broader queries for web search."
    " Just make a judgment on the search query and the search results. Don't do anything else."
    " If you have ordered search coordination three times and the search results are still insufficient,"
    " have the researcher generate 'FINAL ANSWER' based on what you have done so far. Also if the search result is sufficient enough, let the researcher generate 'FINAL ANSWER'"
)

reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Reviewer agent node
def reviewer_agent_node(state):
    reviewer_agent = (reviewer_prompt | llm)
    result=reviewer_agent.invoke(state)
    result = AIMessage(**result.dict(exclude={"type", "name"}), name="Reviewer")

    return {
        "messages": [result]
    }

reviewer_node = functools.partial(reviewer_agent_node)

#Planner
feedback_prompt = (
    "What specific information do you need from the user to conduct better research?"
    "Use given tool to ask key questions and get answers until the user says 'enough'."
    "If the user says enough return with end and get a final confirmation of what you're going to investigate."
    "Do not conduct research."
    "Gather the questions and ask the user at once."
    "If the research topic has been materialized by receiving feedback from the user,"
    "get a final confirmation response from the user by calling UserResponseTool to move on to research. If the user has given additional feedback, correct it and ask again."
)

feedback_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", feedback_prompt ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

feedback_agent=(feedback_agent_prompt | llm)

#Planner 노드 수정
def feedback_node(state):
  while True:
    question = feedback_agent.invoke(state)
    feedback_message = question.content
    print(feedback_message)

    st.chat_message("assistant").write(feedback_message)
    
    user_response = user_input
    user_message = HumanMessage(content=user_response, name="user")

    st.chat_message("user").write(user_message)

    state["messages"].append(feedback_message)
    state["messages"].append(user_message)

    break
      
    pass


#search tool node
from langgraph.prebuilt import ToolNode

tools = [tavily_tool]
tool_node = ToolNode(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

#노드추가
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Visualizer", code_node)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("Planner", feedback_node)
workflow.add_node("search_tool", tool_node)
workflow.add_node("Reviewer",  reviewer_agent_node)


#워크플로우 정의
workflow.add_edge("Visualizer", "supervisor")
workflow.add_edge("Planner", "supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

def research_routing(state):
    last_message = state["messages"][-1]
    if last_message and 'tool_calls' in last_message.additional_kwargs:
        return "search_tool"
    else:
        return "supervisor"

conditional_map_2 = {"supervisor": "supervisor", "search_tool": "search_tool"}
workflow.add_conditional_edges(
    "Researcher",
    research_routing,
    conditional_map_2
)
workflow.add_edge("search_tool", "Reviewer")
workflow.add_edge("Reviewer", "Researcher")


workflow.add_edge(START, "Planner")

graph = workflow.compile()



def ask(query):


    if "messages" in st.session_state:
        st.chat_message("user").write(query)

        # 상태 초기화: messages와 next 필드를 포함한 상태 정의
        state = {
            "messages": [HumanMessage(content=query)],   # HumanMessage를 포함한 메시지 리스트
            "next": "Planner"   # 첫 번째로 실행될 노드를 지정
        }

        response = graph.stream(state)
        print(response)
        parser_callback = AgentCallbacks(AgentState)
        stream_parser = AgentStreamParser(parser_callback)

        for step in response:
            stream_parser.process_agent_steps(step)

            if "__end__" not in step and "Researcher" in step:
            
                researcher_data = step['Researcher']

                if 'messages' in researcher_data:
                    for message in researcher_data['messages']:    
                        st.chat_message("assistant").write(message.content)
                elif 'Planner' in step:
                    feedback_node(state)  # feedback_node 호출

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")  # 사용자 입력 받기

if user_input:
    ask(user_input)  # 사용자 질문 처리
