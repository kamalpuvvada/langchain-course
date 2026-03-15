import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(os.environ.get("AZURE_OPENAI_API_KEY"))
    print(os.environ.get("AZURE_OPENAI_ENDPOINT"))
    print(os.environ.get("AZURE_OPENAI_API_VERSION"))
    print(os.environ.get("AZURE_OPENAI_CHAT_MODEL"))


if __name__ == "__main__":
    main()
