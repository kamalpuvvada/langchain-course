import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

if __name__ == '__main__':

    # pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # index = pc.Index("quickstart")
    print("Ingesting...")
    loader = TextLoader(file_path="mediumblog1.txt", encoding="utf8")
    document = loader.load()

    print("Splitting...")

    text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000)

    texts = text_splitter.split_documents(document)

    print("Embedding...")

    embeddings = AzureOpenAIEmbeddings()
    # vec = embeddings.embed_query("test")
    # print("Embedding dimension:", len(vec))

    print("Ingesting...")

    PineconeVectorStore.from_documents(documents=texts, embedding=embeddings, index_name=os.getenv("INDEX_NAME"))

    print("Done!")