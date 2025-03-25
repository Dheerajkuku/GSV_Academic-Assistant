from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

def main():
    try:
        # Initialize components
        key = "AIzaSyDGHGX6Z7YqNWAFucXZC_gKQzMImoazXLY"
        embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=key, 
            model="models/embedding-001"
        )

        # Load PDF
        path = "/Users/dheerajkumar/Desktop/GSV_Chatbot/GSV DATABASE (1).pdf"
        loader = PyPDFLoader(path)
        documents = loader.load()

        # Create semantic chunks
        print("Creating chunks...")
        semantic_chunker = SemanticChunker(
            embedding_model, 
            breakpoint_threshold_type="percentile"
        )
        semantic_chunks = semantic_chunker.create_documents(
            [d.page_content for d in documents]
        )

        # Create vector store
        print("Creating vector store...")
        Chroma.from_documents(
            documents=semantic_chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db_1"
        )
        
        print("Operation completed successfully!")

    except ImportError as e:
        print(f"Missing dependency: {e}\nPlease install with: pip install {e.name}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()