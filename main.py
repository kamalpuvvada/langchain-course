from dotenv import load_dotenv
import os
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()  # Load environment variables from .env file 
tavily = TavilyClient()

@tool
def search(query:str) -> str:
    """
    Tool that searches over internet
    Args:
        query (str): The search query
    Returns:
        str: The search results
    """
    print(f"Searching for: {query}")
    result = tavily.search(query)
    return result
    # print(f"Search result: {result}")
    # return "Weather I don't know!"

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_MODEL")
    # model_kwargs={"tool_choice": "required"}
)

# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [       
        HumanMessage(content="Find 3 job positions for langchain in Linkedin in Bangalore,India and also check about the weather in Bangalore,India ")]},
        config={"recursion_limt": 3})
    print(result)
    #  SystemMessage(content="Search the tool once for any questions about the weather."
    #     "After tool is used, answer the question based on the tool's output."),

if __name__ == "__main__":  
    main()
