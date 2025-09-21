import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# --- Page Configuration ---
st.set_page_config(
    page_title="Elite Research Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for Light Elite Chat UI ---
def apply_custom_css():
    css = """
    <style>
        /* App background */
        .stApp {
            background-color: #f7f9fc;
            color: #1e1e1e;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Header */
        .main-title {
            text-align: left;
            font-size: 2.2rem;
            font-weight: bold;
            color: #2c3e50;
            padding-top: 10px;
        }

        /* Chat bubbles */
        .user-bubble {
            background-color: #dff7ff;
            padding: 12px 16px;
            border-radius: 16px;
            margin: 8px 0;
            max-width: 75%;
            margin-left: auto;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .bot-bubble {
            background-color: #ffffff;
            padding: 12px 16px;
            border-radius: 16px;
            margin: 8px 0;
            max-width: 75%;
            margin-right: auto;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        /* Input box */
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #1e1e1e;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 10px;
        }

        /* Button */
        .stButton>button {
            background-color: #0078d7;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #005a9e;
        }

        /* Expander for sources */
        .st-expander {
            background-color: #fdfdfd;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .st-expander header {
            color: #1e1e1e;
            font-weight: 500;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_custom_css()

# --- Configuration ---
DATA_PATH = "data/"
DB_PATH = "vector_db/"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_FILE = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

@st.cache_resource(show_spinner=False)
def create_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    if os.path.exists(DB_PATH):
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        loader = DirectoryLoader(
            DATA_PATH, glob="**/*.txt", loader_cls=TextLoader,
            show_progress=False, use_multithreading=True
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    return db

@st.cache_resource(show_spinner=False)
def load_llm():
    if not os.path.exists(LLM_MODEL_FILE):
        st.error(f"LLM model file not found: {LLM_MODEL_FILE}.")
        st.stop()
    llm = CTransformers(
        model=LLM_MODEL_FILE,
        model_type='mistral',
        max_new_tokens=1024,
        temperature=0.7
    )
    return llm

@st.cache_resource(show_spinner=False)
def create_qa_chain(_db, _llm):
    prompt_template = (
        "Use the following context to answer the question. "
        "If you don't know the answer, state that you don't know.\n\n"
        "Context: {context}\nQuestion: {question}\n\nAnswer:"
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    retriever = _db.as_retriever(search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return chain

# --- Sidebar ---
st.sidebar.title("About the Agent 🤖")
st.sidebar.info(
    "Elite Research Agent with a fully local RAG pipeline. "
    "Ask any question, and it will respond using the documents from the `data` folder."
)

# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=80)
with col2:
    st.markdown('<h1 class="main-title">Elite Research Agent</h1>', unsafe_allow_html=True)

st.markdown("---")

# --- Load Components ---
try:
    db = create_vector_db()
    llm = load_llm()
    qa_chain = create_qa_chain(db, llm)
except Exception as e:
    st.error(f"Failed to initialize the agent: {e}")
    st.stop()

# --- Chat History ---
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# --- Display Chat Messages ---
for qa in st.session_state.qa_history:
    st.markdown(f'<div class="user-bubble">{qa["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-bubble">{qa["answer"]}</div>', unsafe_allow_html=True)

    # Sources
    with st.expander("📂 Sources consulted"):
        for doc in qa["sources"]:
            st.write(doc.page_content)

# --- Input for New Question ---
query = st.text_input(
    "Type your question...",
    key=f"query_{len(st.session_state.qa_history)}",
    placeholder="Ask me anything about your documents..."
)

if st.button("Send", key=f"btn_{len(st.session_state.qa_history)}"):
    if query.strip():
        try:
            result = qa_chain.invoke({"query": query})
            st.session_state.qa_history.append({
                "question": query,
                "answer": result["result"],
                "sources": result["source_documents"]
            })
            st.rerun()  # refresh to show the latest chat
        except Exception as e:
            st.error(f"An error occurred: {e}")
