# app.py

import streamlit as st
import google.generativeai as genai
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Use Streamlit's secrets for API key management.
# To use, create a file named `.streamlit/secrets.toml` in your project directory
# with the following content:
# google_api_key = "YOUR_API_KEY_HERE"
# NEVER hardcode your API key in the script.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("API key not found. Please set the 'GEMINI_API_KEY' environment variable or a Streamlit secret.")
    st.stop()

MODEL_NAME = "gemini-2.0-flash-lite"
genai.configure(api_key=API_KEY)

# --- RAG Setup (Simplified In-Memory Vector DB) ---
# For a real application, you would use a persistent vector database like Pinecone,
# Weaviate, or ChromaDB. This example uses a simple in-memory approach with FAISS.

@st.cache_resource
def load_and_index_knowledge_base():
    """
    Loads a predefined knowledge base, creates sentence embeddings, and
    builds a FAISS index for efficient similarity search.
    This function is cached to prevent re-running on every interaction.
    """
    st.info("Initializing knowledge base... This may take a moment.")
    
    # This is our simplified knowledge base about a network.
    # In a real-world scenario, you would ingest documentation files (PDFs, Markdown, etc.).
    network_docs = [
        "Router 'Router-1' is the core router located in the main data center. Its IP address is 10.1.1.1. It handles all North-South traffic.",
        "The primary firewall 'FW-A' is a Palo Alto Networks device. It filters traffic on the WAN uplink and is configured with rules for HTTP, HTTPS, and SSH on standard ports.",
        "Switch 'Switch-2' is a Cisco Catalyst 9300 series switch in the server rack. It connects all physical servers via VLAN 10 (192.168.10.0/24).",
        "The wireless access point network is managed by a centralized controller at 10.1.20.10. It broadcasts SSIDs 'Guest-Net' and 'Corp-Net'.",
        "For change management, all changes must be submitted via the Jira ticket system at https://jira.company.com. An approval from network engineering lead is required.",
        "The network's primary DNS server is 10.1.1.5, and the secondary is 10.1.1.6.",
        "A network diagram is available on the internal SharePoint site under 'Network_Diagrams/Current_Topology.vsdx'.",
    ]

    # Load a pre-trained Sentence Transformer model for creating embeddings.
    # 'all-MiniLM-L6-v2' is a small, efficient model.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for all network documents.
    embeddings = model.encode(network_docs)
    
    # FAISS requires numpy arrays of float32.
    embeddings = np.array(embeddings).astype('float32')
    
    # Get the dimension of the embeddings.
    d = embeddings.shape[1]
    
    # Build a FAISS index (IndexFlatL2 is a simple brute-force index).
    index = faiss.IndexFlatL2(d)
    
    # Add the embeddings to the index.
    index.add(embeddings)
    
    st.success("Knowledge base loaded and indexed!")
    return model, index, network_docs

def retrieve_context(query, model, index, docs, k=2):
    """
    Performs a similarity search on the FAISS index to find the most relevant
    documents for a given query.
    """
    # Create an embedding for the user's query.
    query_embedding = model.encode([query]).astype('float32')
    
    # Perform a search for the top 'k' most similar documents.
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the content of the relevant documents.
    relevant_docs = [docs[i] for i in indices[0]]
    
    return "\n".join(relevant_docs)

# --- Streamlit UI ---
st.set_page_config(page_title="Network Documentation Assistant", layout="wide")

st.title("🧠 MANISH -  GenAI Network Documentation Assistant")
st.markdown(
    """
    This assistant helps you query and manage network documentation, change plans,
    and a knowledge base. It uses a Retrieval-Augmented Generation (RAG)
    approach to provide context-aware answers.
    """
)

# Load the RAG components.
st_model, st_index, st_docs = load_and_index_knowledge_base()

st.header("Ask a Question about the Network")
user_query = st.text_input(
    "Enter your query here:",
    placeholder="e.g., What is the IP address of the core router? or How do I request a network change?"
)

if st.button("Generate Response"):
    if not user_query:
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching and generating response..."):
                # 1. Retrieve relevant context from the knowledge base (RAG step).
                context = retrieve_context(user_query, st_model, st_index, st_docs)
                
                # 2. Construct the RAG prompt.
                # This prompt includes the context from the knowledge base.
                augmented_prompt = (
                    f"You are a network documentation assistant. Use the following "
                    f"network knowledge to answer the user's question. If the information "
                    f"is not available in the provided context, state that you cannot answer.\n\n"
                    f"Network Knowledge:\n---\n{context}\n---\n\n"
                    f"User's Question: {user_query}\n\n"
                    f"Your Answer:"
                )

                # 3. Call the Gemini API.
                # Note: The `gemini-2.0-flash-lite` model is a fast, efficient model.
                model_client = genai.GenerativeModel(MODEL_NAME)
                response = model_client.generate_content(augmented_prompt)
                
                # 4. Display the response.
                st.subheader("Assistant's Answer")
                st.success(response.text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please ensure your API key is correct and try again.")


# --- Additional notes and disclaimers ---
st.markdown("---")
st.info(
    """
    **Deployment Notes:**
    * **Vector Database:** This example uses a simplified in-memory FAISS index. For a production
      environment, you would use a persistent, scalable vector database.
    * **Finetuning:** Finetuning a model is a separate process of training it on a specific
      dataset. It's a complex task and not something that can be included in a simple
      Streamlit app file. The RAG approach serves as a powerful alternative for
      specializing the model's knowledge on-the-fly.
    * **Security:** Your API key is managed securely using Streamlit's `secrets.toml`
      feature.
    """
)
