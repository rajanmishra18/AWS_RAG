from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 1. Load document
loader = TextLoader("data/notes.txt")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 3. Create embeddings (FREE)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 4. Build vector store (in memory)
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Ask a question
query = "What is RAG?"

docs = vectorstore.similarity_search(query, k=2)
context = "\n".join([doc.page_content for doc in docs])

# 6. Send to local LLM
llm = Ollama(model="llama3")

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

answer = llm.invoke(prompt)
print("\nAnswer:\n", answer)
