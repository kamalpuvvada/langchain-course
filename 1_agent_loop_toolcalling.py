from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langchain.chat_models import init_chat_model
load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product_name: str) -> float:
    """Simulate fetching the price of a product."""
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99
    }
    print(f"Fetching price for {product_name}...")
    return prices.get(product_name.lower(), 0.0)

@tool
def apply_discount(tier:str, price: float) -> float:
    """Simulate applying a discount based on the tier."""
    discounts = {
        "bronze": 5,
        "silver": 10,
        "gold": 15
    }
    print(f"Applying {tier} discount to price {price}...")
    discount_percent = discounts.get(tier.lower(), 0.0)
    return round(price * (1 - discount_percent)/100, 2)

@traceable(name="Lang chain agent loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(
            model=f"ollama:{MODEL}",
            temperature=0
        )
    llm_with_tools = llm.bind_tools(tools)   
    messages = [
        SystemMessage(
            content=
            "You are a helpful shopping assistant. "
            "You have access to a product catalog tool "
            "and a discount tool.\n\n"
            "STRICT RULES — you must follow these exactly:\n"
            "1. NEVER guess or assume any product price. "
            "You MUST call get_product_price first to get the real price.\n"
            "2. Only call apply_discount AFTER you have received "
            "a price from get_product_price — do NOT pass a made-up number.\n"
            "3. NEVER calculate discounts yourself using math. "
            "Always use the apply_discount tool.\n"
            "4. If the user does not specify a discount tier, "
            "ask them which tier to use — do NOT assume one."
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        ai_message = llm_with_tools.invoke(messages)
        print(f"ai_message: {ai_message}")
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("No tools called. Ending loop.")
            print("Final Answer:", ai_message.content)
            return ai_message.content

        # Run the first tool call and get the result
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id = tool_call.get("id")
        print(f"Tool called: {tool_name} with args {tool_args}")
        tool_func = tools_dict.get(tool_name)
        if tool_func is None:
            print(f"Unknown tool: {tool_name}. Ending loop.")
            return
        
        observation = tool_func.invoke(tool_args)
        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=observation,tool_call_id=tool_id))

    print("Max iterations reached. Ending loop.")
    return
        
if __name__ == "__main__":
    print("Welcome to the Product Price Agent!")
    run_agent("What is the price of a laptop with a gold tier?")