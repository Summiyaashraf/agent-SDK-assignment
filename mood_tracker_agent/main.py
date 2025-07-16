from agents import Runner, Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import os
import chainlit as cl
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()

gemini_api_key=os.getenv("GEMINI_API_KEY")

# step 01
external_client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# step 02
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
# step 03
config=RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)
# step 04
#  Mood Detector Agent
mood_agent = Agent(
    name="Mood Detector",
    instructions="""
You are a mood detection assistant.

→ Your job is to detect the user's mood **only when they describe how they feel** (e.g. "I’m feeling sad", "I’m stressed").

→ If the user just says hello, tells their name, or says anything unrelated to emotions — respond only with: "unknown"

Valid moods: happy, sad, stressed, angry, excited, tired, anxious, etc.

Your final reply should be only **one lowercase word** — no extra explanation.
"""
)

# Mood suggestion

suggestion_agent = Agent(
    name="Mood Helper",
    instructions="""
If the user's mood is sad or stressed, suggest a relaxing or fun activity to improve their mood.
If the mood is anything else, say: 'You're doing great! No suggestions needed.'
"""
)

# Mood Router
router_agent = Agent(
    name="Mood Router",
    instructions="""
You are a smart emotional assistant. Your job is to understand the user's mood and decide what to do:

- If the mood is sad or stressed → hand off to the Mood Helper agent.
- If the mood is happy or anything else → respond with just the mood.

Only hand off when mood needs help.
""",
    handoffs=[mood_agent,suggestion_agent]
)

# step 5
@cl.on_chat_start
async def handle_start():
     cl.user_session.set("history", [])
     await cl.Message(content="Hello From Summiya Ashraf! How can help you today about your mood?").send()
# step 06
@cl.on_message
async def handle_message(message: cl.Message):

    history = cl.user_session.get("history")

    history.append({"role":"user", "content": message.content})

# step 07
    msg= cl.Message(content="")
    await msg.send()

    result= Runner.run_streamed(
         router_agent,
         input=history,
         run_config=config
    )

    # step 8
    async for event in result.stream_events():
         if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
              await msg.stream_token(event.data.delta)

    # step 09
    history.append({"role":"assistant", "content": result.final_output})
    history= cl.user_session.set("history",history)

