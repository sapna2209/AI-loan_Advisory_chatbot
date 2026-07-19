import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

print("Loading PDFs...")

loader = PyPDFDirectoryLoader("data")
documents = loader.load()

print("\n========== Loaded Pages ==========\n")

for doc in documents:
    print(f"File : {os.path.basename(doc.metadata['source'])}")
    print(f"Page : {doc.metadata['page'] + 1}")
    print(f"Characters Extracted : {len(doc.page_content)}")

    # Print first 200 characters
    preview = doc.page_content[:200].replace("\n", " ")
    print("Preview :", preview)
    print("-" * 80)

print("==================================")
print("Total Pages:", len(documents))

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"\nCreated {len(chunks)} chunks.")

print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embedding=embeddings
)

vectorstore.save_local("vector_db")

print("\n✅ Vector database created successfully!")