import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()

print("Initializing...")

embeddings = AzureOpenAIEmbeddings()
llm = AzureChatOpenAI(model=os.getenv("AZURE_OPENAI_CHAT_MODEL"))
vector_store = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    """ Answer the question based on the following context.
    {context}
    Question: {question}
    Provide a detailed answer. """
)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def without_lecl(query: str):
    """
    Simple retreival chain without lecl. We will just retrieve the documents and print them out. 
    This is to show the difference between using lecl and not using lecl.

    Limitations:
    Manual step by step process. 
    No built in streaming support.
    No aync support.
    Harder to compose with other chains.
    More verbose and error prone.
    """
    print("Retrieving without using lecl...")
    documents = retriever.invoke(query)
    formatted_docs = format_docs(documents)
    prompt_tosend = prompt.format_prompt(context=formatted_docs, question=query).to_messages()
    response = llm.invoke(prompt_tosend)
    print(f"Answer: {response.content}")
    return response.content
    

def no_rag(query: str):
        print("No rag implementation:")
        print("==" * 20)

        messages = [HumanMessage(content=query)]
        answer = llm.invoke(messages)
        print(f"Answer: {answer.content}")
        return answer.content

def lecl():
    """
    Lecl implementation. We will use the lecl chain to retrieve the documents and generate the answer. 
    This is to show the benefits of using lecl.

    Benefits:
    Built in streaming support.
    Async support.
    Easier to compose with other chains.
    Less verbose and less error prone.
    """
    # retreival_chain =(
    #     retreiver |
    #     format_docs |
    #     prompt |
    #     llm |
    #     StrOutputParser()
    # )

    # problem with above implementation is that we cannot pass context and question separately to the prompt. 
    # We need to combine them into a single string and pass it to the prompt. 
    # This is not ideal and can lead to errors if the formatting is not correct.
    # So we need to RunnablePassthrough to pass the context and question separately to the prompt.

    print("Lecl implementation:")

    retreival_chain =(
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs,  
        ) |  # Pass the input query as is to the retriever
        # Pass the retrieved documents as context to the prompt
        prompt |  # The prompt will take care of formatting the context and question
        llm |
        StrOutputParser()
    )
    return retreival_chain


if __name__ == '__main__':
    question = "What is pinecone in machine learning?"

    # no_rag(question)

    # without_lecl(question)

    lecl_chain = lecl()
    response = lecl_chain.invoke({"question": question})
    print(f"Answer: {response}")


    # print("Retrieving...")
    # docs = retriever._get_relevant_documents(query=question)
    # print("Generating answer...")
    # humanMessage = HumanMessage(content=prompt.format_prompt(context=format_docs(docs), question=question).to_messages())
    # answer = llm.invoke(humanMessage)
    # print("Answer:")
    # print(answer)