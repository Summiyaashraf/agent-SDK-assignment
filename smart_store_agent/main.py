from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()
gemini_api_key= os.getenv("GEMINI_API_KEY")

# step 01
external_client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# step 02
model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash",
    openai_client=external_client
)

# step 03
config= RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled= True
)

# step 04
agent = Agent(
    name="smart_store_agent",
    instructions="You are a smart store assistant. Based on the user's symptoms or needs, suggest a product (like a medicine) and explain clearly why it is helpful."
)

# step 05
@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello, How can I help you today?").send()

# step 06
@cl.on_message
async def handle_message(message : cl.Message):
    history= cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

# step 07
    msg= cl.Message(content="")
    await msg.send()

    result= Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )
# step 08

    async for event in result.stream_events():

        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

# step 09
    history.append({"role":"assistant", "content": result.final_output})
    history= cl.user_session.set("history", history)
   
