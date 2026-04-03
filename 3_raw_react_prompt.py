from dotenv import load_dotenv
from langsmith import traceable
from ollama import chat
import inspect
import json
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
    return round(price - (price * (discount_percent)/100), 2)

tools={
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tools_description(tools_dict):
    descriptions = []
    for tool_name, tool_func in tools_dict.items():
        original_function = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_function)
        doc_string = inspect.getdoc(original_function) or "No description available."
        descriptions.append(f"{tool_name}{signature}: {doc_string}")
    return "\n".join(map(str, descriptions))


tools_description = get_tools_description(tools)
tools_names = ", ".join(tools.keys())
react_prompt = """
Strict_rules:
1. NEVER guess or assume any product price. 
You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received
a price from get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math.
Always use the apply_discount tool.
4. If the user does not specify a discount tier,
ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:
{tools_description}
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:"""

@traceable(name="Ollama chat trace", run_type="llm")
def ollama_chat_trace(model,messages,options):
    llm = chat(model=f"{model}",messages=messages, options=options)
    return llm

import re

def parse_action_block(text: str):
    action_match = re.search(r"(?mi)^\s*Action:\s*(.+?)\s*$", text)
    input_match = re.search(
        r"(?mis)^\s*Action Input:\s*(.+?)\s*(?=^\s*(Thought:|Action:|Observation:|Final Answer:)|\Z)",
        text,
    )

    action = action_match.group(1).strip() if action_match else None
    action_input = input_match.group(1).strip() if input_match else None

    # Optional cleanup for common formatting from LLMs
    if action:
        action = action.strip("`\"' ")
    if action_input:
        action_input = action_input.strip("`\"' ")

    return action, action_input

@traceable(name="Lang chain agent loop")
def run_agent(question:str):

    print(f"Question: {question}")
    print("=" * 50)

    prompt = react_prompt.format(tools_description=tools_description,tools_names=tools_names, input=question)
    scratchpad = ""
   
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + "\n" + scratchpad

        messages = [
            { "role": "user", "content": full_prompt }
        ]
        options = {"temperature": 0, "stop": ["\nObservation"]}
        
        ai_message = ollama_chat_trace(MODEL, messages, options)
        output = ai_message.message.content
        print(f"ai_message: {ai_message}")
        action = re.search(r"Action:\s*(.+?)$", output, re.MULTILINE)
        if action:
            action = action.group(1).strip()
            print(f"Action found: {action}")

        action_input = re.search(r"Action Input:\s*(.+?)$", output, re.MULTILINE)
        if action_input:
            action_input = action_input.group(1).strip()
            print(f"Action Input found: {action_input}")
        
        if not action:
            print("No action found. Ending loop.")
            final_answer=re.search(r"Final Answer:\s*(.+)$", output, re.MULTILINE)
            if final_answer:
                print("Final Answer:", final_answer.group(1))
            return final_answer.group(1) 

        # Run the first tool call and get the result
        # tool_id = tool_call.id
        print(f"Tool called: {action} with args {action_input}")
        tool_func = tools.get(action)
        if tool_func is None:
            print(f"Unknown tool: {action}. Ending loop.")
            return
        
        try:
            action_input_json = json.loads(action_input)
            if isinstance(action_input_json, dict):
                action_input = action_input_json
        except json.JSONDecodeError:
            pass

        if isinstance(action_input, dict):
            observation = tool_func(**action_input)
        else:
            observation = tool_func(action_input)
        print(f"Observation: {observation}")

        scratchpad = "\n".join([scratchpad, output, f"Observation: {observation}"])

    print("Max iterations reached. Ending loop.")
    return
        
if __name__ == "__main__":
    print("Welcome to the Product Price Agent!")
    run_agent("What is the price of a laptop with a gold tier?")