from typing import Literal, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import chainlit as cl
from chainlit.element import Element
from io import BytesIO
from openai import AsyncOpenAI

from actions_test import Go2Action
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def create_tool_agent(model, tools):
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]

        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: MessagesState):
        messages = state['messages']
        response = model.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    
    tool_node = ToolNode(tools)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", 'agent')

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    return app


#ChannelFactoryInitialize(0, "enp0s31f6")
action = Go2Action()

@tool
def StandUp():
    """立ち上がる"""
    action.StandUp()

@tool
def SitDown():
    """座る"""
    action.SitDown()

@tool
def Stretch():
    """ストレッチする"""
    action.Stretch()

@tool
def Dance():
    """ダンスする"""
    action.Dance()

@tool
def FrontJump():
    """前方にジャンプする"""
    action.FrontJunmp()

@tool
def Heart():
    """前脚でハートを描く"""
    action.Heart()

@tool
def FrontFlip():
    """バク転する"""
    action.FrontFlip()

@tool
def Move(x: float, y: float, z: float):
    """前方にx(m)、右にy(m)移動し、z(rad)半時計回りに回転する。後退する場合は-x(m)、左に移動する場合は-y(m)、時計回りに回転する場合は-z(rad)というように指定する。"""
    action.Move(x, y, z)


tools = [StandUp, SitDown, Stretch, Dance, Heart, FrontFlip, Move]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
app = create_tool_agent(model, tools)


async_openai_client = AsyncOpenAI()

@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: List[Element]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()
    
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    msg = cl.Message(author="You", content=transcription, elements=elements)

    await main(message=msg)


@cl.on_message
async def main(message):
    final_state = app.invoke(
        {"messages": [HumanMessage(content=message.content)]},
        config={"configurable": {"thread_id": 42}}
    )
    response = final_state["messages"][-1].content

    await cl.Message(
        content=response
    ).send()
