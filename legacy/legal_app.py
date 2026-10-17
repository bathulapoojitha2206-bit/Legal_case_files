import streamlit as st
import numpy as np
import faiss
import ollama

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Legal Document RAG",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Document RAG Assistant")
st.write("Ask questions from your legal documents.")

# --------------------------------------------------
# LOAD FAISS INDEX
# --------------------------------------------------

@st.cache_resource
def load_index():
    return faiss.read_index("legal.index")


index = load_index()

# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

@st.cache_resource
def load_chunks():
    import pickle

    with open("chunks.pkl", "rb") as f:
        return pickle.load(f)


chunks = load_chunks()

# --------------------------------------------------
# EMBEDDING FUNCTION
# --------------------------------------------------

def embed_question(question):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )

    question_embedding = np.array(
        [response["embedding"]],
        dtype="float32"
    )

    return question_embedding


# --------------------------------------------------
# FAISS SEARCH
# --------------------------------------------------

def search_faiss(question_embedding, index, k=3):

    distances, indices = index.search(
        question_embedding,
        k
    )

    retrieved_chunks = []

    for i in indices[0]:

        retrieved_chunks.append(
            chunks[i]
        )

    return distances[0], indices[0], retrieved_chunks


# --------------------------------------------------
# GENERATE ANSWER USING OLLAMA
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):

    context = ""

    for chunk in retrieved_chunks:

        context += (
            "\nSource: "
            + chunk["source"]
            + "\n"
            + chunk["text"]
            + "\n"
        )

    prompt = f"""
You are a legal document assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context,
say "The information is not available in the retrieved documents."

Do not make up legal information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.generate(
        model="llama3.2",
        prompt=prompt
    )

    return response["response"]


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

question = st.text_input(
    "Enter your legal question:"
)


# --------------------------------------------------
# SEARCH
# --------------------------------------------------

if st.button("Ask Question"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents..."):

            # Create embedding
            question_embedding = embed_question(
                question
            )

            # Search FAISS
            distances, indices, retrieved_chunks = search_faiss(
                question_embedding,
                index,
                k=3
            )

            # Generate answer
            answer = generate_answer(
                question,
                retrieved_chunks
            )

        # --------------------------------------------------
        # DISPLAY ANSWER
        # --------------------------------------------------

        st.subheader("Answer")

        st.write(answer)

        # --------------------------------------------------
        # DISPLAY SOURCES
        # --------------------------------------------------

        st.subheader("Retrieved Sources")

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            with st.expander(
                f"Source {i}: {chunk['source']}"
            ):

                st.write(
                    chunk["text"]
                )

        # --------------------------------------------------
        # RETRIEVAL DETAILS
        # --------------------------------------------------

        st.subheader("Retrieval Details")

        st.write(
            "Number of retrieved chunks:",
            len(retrieved_chunks)
        )

        st.write(
            "FAISS indices:",
            indices.tolist()
        )

        st.write(
            "Distances:",
            distances.tolist()
        )