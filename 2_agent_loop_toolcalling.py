from dotenv import load_dotenv
from langsmith import traceable
from ollama import chat
load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """Simulate fetching the price of a product."""
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99
    }
    print(f"Fetching price for {product_name}...")
    return prices.get(product_name.lower(), 0.0)

@traceable(run_type="tool")
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

llm_tools = [
    {
        "type":"function",
        "function":{
            "name": "get_product_price",
            "description": "Fetch the price of a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product to fetch the price for."
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name": "apply_discount",
            "description": "Apply a discount to a price based on the tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tier": {
                        "type": "string",
                        "description": "The discount tier (bronze, silver, gold)."
                    },
                    "price": {
                        "type": "number",
                        "description": "The original price to apply the discount to."
                    }
                },
                "required": ["tier", "price"]
            }
        }
    }
]

@traceable(name="Ollama chat trace", run_type="llm")
def ollama_chat_trace(messages):
    llm = chat(model=f"{MODEL}",tools=llm_tools, messages=messages)
    return llm


@traceable(name="Lang chain agent loop")
def run_agent(question:str):
    tools_dict={
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }
    messages = [
        { "role": "system", "content":
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
        },
        { "role": "user", "content": question }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        ai_message = ollama_chat_trace(messages)
        print(f"ai_message: {ai_message}")
        tool_calls = ai_message.message.tool_calls

        if not tool_calls:
            print("No tools called. Ending loop.")
            print("Final Answer:", ai_message.message)
            return ai_message.message

        # Run the first tool call and get the result
        tool_call = tool_calls[0]
        tool_name = tool_call.function.get("name")
        tool_args = tool_call.function.get("arguments")
        # tool_id = tool_call.id
        print(f"Tool called: {tool_name} with args {tool_args}")
        tool_func = tools_dict.get(tool_name)
        if tool_func is None:
            print(f"Unknown tool: {tool_name}. Ending loop.")
            return
        
        observation = tool_func(**tool_args)
        print(f"Observation: {observation}")

        messages.append(ai_message.message)
        tool_message = {
            "role": "tool",
            "content": str(observation)
        }
        messages.append(tool_message)

    print("Max iterations reached. Ending loop.")
    return
        
if __name__ == "__main__":
    print("Welcome to the Product Price Agent!")
    run_agent("What is the price of a laptop with a gold tier?")