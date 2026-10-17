import faiss

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder


# ==========================================
# 1. LOAD DOCUMENTS
# ==========================================

DATA_PATH = "langchain"

loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

docs = loader.load()

print("Documents loaded:", len(docs))


# ==========================================
# 2. SPLIT DOCUMENTS
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = splitter.split_documents(docs)

print("Chunks created:", len(split_docs))


# Add chunk IDs
for i, doc in enumerate(split_docs):
    doc.metadata["chunk_id"] = i


# ==========================================
# 3. CREATE EMBEDDINGS
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 4. CREATE FAISS VECTOR STORE
# ==========================================

vector_store = FAISS.from_documents(
    split_docs,
    embeddings
)

print("FAISS vector store created successfully!")


# ==========================================
# 5. RETRIEVER
# ==========================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)


# ==========================================
# 6. OLLAMA LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2"
)


# ==========================================
# 7. PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template("""
You are a factual assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not found in the context, say:
  Not found in the document.

Context:
{context}

Question:
{question}

Answer:
""")


# ==========================================
# 8. CROSS-ENCODER RERANKER
# ==========================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==========================================
# 9. ASK FUNCTION
# ==========================================

def ask(query: str):

    # Retrieve initial documents
    initial_docs = retriever.invoke(query)

    # Create query-document pairs
    pairs = [
        (query, doc.page_content)
        for doc in initial_docs
    ]

    # Calculate reranking scores
    scores = reranker.predict(pairs)

    # Sort by score
    ranked = sorted(
        zip(scores, initial_docs),
        key=lambda x: x[0],
        reverse=True
    )

    # Take top 3
    final_docs = [
        doc
        for _, doc in ranked[:3]
    ]

    # Create context
    context = "\n\n".join(
        doc.page_content
        for doc in final_docs
    )

    # Ask LLM
    response = llm.invoke(
        prompt.format(
            context=context,
            question=query
        )
    )

    return {
        "answer": response.content,

        "sources": [
            {
                "file": doc.metadata.get("source"),
                "chunk": doc.metadata.get("chunk_id")
            }
            for doc in final_docs
        ]
    }


# ==========================================
# 10. TEST
# ==========================================

question = input("Enter your question: ")

result = ask(question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(source)