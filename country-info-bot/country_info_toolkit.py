from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
import os 
from dotenv import load_dotenv
import chainlit as cl
import requests

# Step 01: Load .env & Setup API Key
load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 02: Provider & Model Setup
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# Step 03: Country Info Tools

@function_tool
def get_capital(country: str) -> str:
    """
    Returns the capital of the given country using REST API.
    """
    try:
        res = requests.get(f"https://restcountries.com/v3.1/name/{country}")
        data = res.json()
        capital = data[0]["capital"][0]
        return f"The capital of {country.title()} is {capital}."
    except Exception as e:
        return f"❌ Could not fetch capital: {e}"

@function_tool
def get_language(country: str) -> str:
    """
    Returns the main language(s) of the given country using REST API.
    """
    try:
        res = requests.get(f"https://restcountries.com/v3.1/name/{country}")
        data = res.json()
        languages = list(data[0]["languages"].values())
        return f"The main language(s) of {country.title()} are: {', '.join(languages)}."
    except Exception as e:
        return f"❌ Could not fetch language: {e}"

@function_tool
def get_population(country: str) -> str:
    """
    Returns the population of the given country using REST API.
    """
    try:
        res = requests.get(f"https://restcountries.com/v3.1/name/{country}")
        data = res.json()
        population = data[0]["population"]
        return f"The population of {country.title()} is approximately {population:,}."
    except Exception as e:
        return f"❌ Could not fetch population: {e}"

# Step 04: Main Orchestrator Agent
agent = Agent(
    name="Country Info Assistant",
    instructions="""
You are a helpful assistant that provides information about countries only.

When the user gives a country name, call:
- get_capital
- get_language
- get_population

If the user asks something unrelated (like greeting, joke), do not respond directly — the developer will handle that.
""",
    model=model,
    tools=[get_capital, get_language, get_population]
)

# Step 05: Chainlit Setup

@cl.on_chat_start
async def start():
    await cl.Message(content="🌍 Hello! I'm your Country Info Bot.\nType a country name to know its capital, language, and population.").send()

@cl.on_message
async def respond(message: cl.Message):
    user_input = message.content.strip()
    lower_input = user_input.lower()

    # Case 1: Greeting
    if any(greet in lower_input for greet in ["hi", "hello", "hey", "salam", "assalamualaikum"]):
        await cl.Message(content="👋 Hello! I'm your Country Info Bot.\nPlease type a country name you'd like to know about.").send()
        return

    # Case 2: Name Introduction
    if "my name is" in lower_input or "i am" in lower_input:
        name = lower_input.replace("my name is", "").replace("i am", "").strip().title()
        if len(name.split()) == 1:
            await cl.Message(content=f"😊 Nice to meet you, {name}!\nPlease type a country name to continue.").send()
        else:
            await cl.Message(content="😊 Nice to meet you!\nPlease type a country name to get started.").send()
        return

    # Case 3: Invalid Short Input
    if len(user_input) < 3:
        await cl.Message(content="❗ Please enter a valid country name like `Pakistan`, `USA`, or `India`.").send()
        return

    # Case 4: Run Agent for Country Info
    await cl.Message(content="🔎 Fetching data from live API...").send()

    result = Runner.run_sync(
        agent,
        input=f"Tell me about {user_input}"
    )

    await cl.Message(content=f"✅ {result.final_output}").send()
