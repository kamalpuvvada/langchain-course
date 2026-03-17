import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    # print(os.environ.get("AZURE_OPENAI_API_KEY"))
    # print(os.environ.get("AZURE_OPENAI_ENDPOINT"))
    # print(os.environ.get("AZURE_OPENAI_API_VERSION"))
    # print(os.environ.get("AZURE_OPENAI_CHAT_MODEL"))

    information = """
    **Elon Musk is a South African born entrepreneur best known as the CEO of Tesla and SpaceX, and one of the wealthiest people in the world. He has founded or co-founded several influential companies, including PayPal, Neuralink, and xAI.**  [en.wikipedia.org](https://en.wikipedia.org/wiki/Elon_Musk)  [en.wikipedia.org](https://en.wikipedia.org/wiki/Elon_Musk)  [Forbes](https://www.forbes.com/profile/elon-musk/)  

        ### Key Facts About Elon Musk
        - **Full Name:** Elon Reeve Musk  
        - **Date of Birth:** June 28, 1971  
        - **Place of Birth:** Pretoria, South Africa  
        - **Citizenship:** South Africa, Canada, United States  
        - **Education:** University of Pennsylvania (BA in Economics, BS in Physics)  
        - **Current Roles:**  
        - CEO and product architect of **Tesla**  
        - Founder, CEO, and chief engineer of **SpaceX**  
        - Founder and CEO of **xAI**  
        - Founder of **The Boring Company** and **X Corp.**  
        - Co-founder of **Neuralink**, **OpenAI**, **Zip2**, and **X.com** (which became PayPal)  [en.wikipedia.org](https://en.wikipedia.org/wiki/Elon_Musk)  

        ### Personal Life
        - **Parents:** Errol Musk (father), Maye Musk (mother)  
        - **Spouses:** Justine Wilson (2000–2008), Talulah Riley (2010–2012, 2013–2016)  
        - **Children:** 14 publicly known, including Vivian Wilson  [en.wikipedia.org](https://en.wikipedia.org/wiki/Elon_Musk)  

        ### Wealth & Influence
        - **Net Worth (Feb 2026):** Estimated at **US$852 billion**, making him the richest person globally.  
        - Musk owns significant stakes in Tesla and SpaceX, and his ventures have reshaped industries from electric vehicles to private space exploration.  [en.wikipedia.org](https://en.wikipedia.org/wiki/Elon_Musk)  [Forbes](https://www.forbes.com/profile/elon-musk/)  

        ### Impact
        - **Tesla:** Revolutionized electric cars and clean energy adoption.  
        - **SpaceX:** Advanced reusable rockets and aims for Mars colonization.  
        - **xAI & Neuralink:** Exploring artificial intelligence and brain–computer interfaces.  
        - **X Corp. (formerly Twitter):** Acquired in 2022, merged with xAI in 2025.  [Forbes](https://www.forbes.com/profile/elon-musk/)  
    """

    summary_template = """
    Summarize the following information {information} and give me 3 key points
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = AzureChatOpenAI(
        # azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        # openai_api_version=os.environ.get("AzURE_OPENAI_API_VERSION"),
        # openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_MODEL")
    )

    # llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm
    result = chain.invoke(input={"information": information})

    print(result.content)


if __name__ == "__main__":
    main()
