import os, io, json, fitz, matplotlib.pyplot as plt, streamlit as st
from typing import List, Dict, Any
from scipy.special import softmax
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from fpdf import FPDF

# LangChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------------- CONFIG ----------------
FINBERT_MODEL = "ProsusAI/finbert"
INSTRUCTION_MODEL = "google/flan-t5-large"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_dynamic_fin"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    tokenizer_finbert = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model_finbert = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(DEVICE)
    tokenizer_t5 = AutoTokenizer.from_pretrained(INSTRUCTION_MODEL)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(INSTRUCTION_MODEL).to(DEVICE)
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return tokenizer_finbert, model_finbert, tokenizer_t5, model_t5, embedding_fn

tokenizer_finbert, model_finbert, tokenizer_t5, model_t5, embedding_fn = load_models()

# ---------------- UTILITIES ----------------
def extract_text(file) -> str:
    if file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text("text") for page in pdf])
    return file.read().decode("utf-8")

def finbert_sentiment(text: str) -> Dict[str, float]:
    inputs = tokenizer_finbert(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = model_finbert(**inputs).logits[0].cpu().numpy()
    probs = softmax(logits)
    labels = ["positive", "negative", "neutral"]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

def init_vectorstore(embedding_fn) -> Chroma:
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_fn)

def add_document(store: Chroma, text: str, metadata: dict = None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents([Document(page_content=text, metadata=metadata or {})])
    store.add_documents(chunks)
    store.persist()

def retrieve_context(store: Chroma, query: str, k=3) -> str:
    retriever = store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)
    return "\n\n".join([r.page_content for r in results])

def generate_structured_summary(text: str, context: str, sentiment: Dict[str, float]) -> Dict[str, Any]:
    text, context = text[:1500], context[:1000]
    prompt = f"""
Summarize the REPORT into JSON.

Schema:
{{
  "summary": "One-sentence summary.",
  "drivers": ["2-3 key drivers"],
  "risks": ["2-3 key risks"],
  "recommendation": "Buy / Hold / Sell",
  "confidence": 0.0-1.0
}}

CONTEXT:
{context}

REPORT:
{text}

FINBERT SENTIMENT:
{json.dumps(sentiment, indent=2)}
Return only JSON.
"""
    inputs = tokenizer_t5(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model_t5.generate(**inputs, max_length=512, num_beams=5, temperature=0.4)
    decoded = tokenizer_t5.decode(outputs[0], skip_special_tokens=True).strip()
    try:
        json_str = decoded[decoded.find("{"): decoded.rfind("}") + 1]
        return json.loads(json_str)
    except Exception:
        return {"error": "Failed to parse JSON", "raw_output": decoded}

def generate_abstractive_summary(text: str, context: str, sentiment: Dict[str, float]) -> str:
    prompt = f"""
Write a fluent 2–3 sentence summary using CONTEXT and SENTIMENT.

CONTEXT:
{context}

REPORT:
{text}

FINBERT SENTIMENT:
{json.dumps(sentiment, indent=2)}
"""
    inputs = tokenizer_t5(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model_t5.generate(**inputs, max_length=180, num_beams=4, temperature=0.7)
    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True).strip()

def compute_sentiment_trend(texts: List[str]) -> Dict[str, List[float]]:
    pos, neg, neu = [], [], []
    for t in texts:
        s = finbert_sentiment(t)
        pos.append(s["positive"]); neg.append(s["negative"]); neu.append(s["neutral"])
    return {"positive": pos, "negative": neg, "neutral": neu}

def plot_sentiment_trend(trend: Dict[str, List[float]]):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(1, len(next(iter(trend.values()))) + 1)
    for k, v in trend.items():
        ax.plot(x, v, marker="o", label=k.capitalize())
    ax.set_title("📈 Sentiment Trend Over Time")
    ax.set_xlabel("Report Index")
    ax.set_ylabel("Sentiment Score")
    ax.legend(); ax.grid(alpha=0.3)
    return fig

def export_pdf(summary_json, abstract_summary, sentiment, fig):
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Financial Report Summary", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"\nSentiment:\n{json.dumps(sentiment, indent=2)}")
    pdf.multi_cell(0, 10, f"\nStructured Summary:\n{json.dumps(summary_json, indent=2)}")
    pdf.multi_cell(0, 10, f"\nAbstractive Summary:\n{abstract_summary}")
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
    img_path = "temp_trend.png"
    with open(img_path, "wb") as f: f.write(buf.read())
    pdf.image(img_path, x=10, w=180)
    os.remove(img_path)
    return pdf.output(dest="S").encode("latin-1")

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Dynamic Financial AI", layout="wide")
st.title("💼 Dynamic Generative AI Financial Report Summarization")

uploaded_file = st.sidebar.file_uploader("📂 Upload Report (.pdf/.txt)", type=["pdf", "txt"])
store = init_vectorstore(embedding_fn)

if uploaded_file:
    report_text = extract_text(uploaded_file)
    company_name = uploaded_file.name.split(".")[0]
    st.subheader(f"📊 Analyzing: {company_name}")
    st.text_area("Extracted Text", report_text[:2000] + "...", height=200)

    # Add to RAG memory
    add_document(store, report_text, {"filename": company_name})
    st.success(f"✅ Added {company_name} to RAG memory!")

    sentiment = finbert_sentiment(report_text)
    st.subheader("FinBERT Sentiment"); st.json(sentiment)

    context = retrieve_context(store, report_text)
    st.subheader("RAG Context"); st.text_area("Relevant Context", context, height=200)

    summary_json = generate_structured_summary(report_text, context, sentiment)
    st.subheader("Structured Summary"); st.json(summary_json)

    abstract_summary = generate_abstractive_summary(report_text, context, sentiment)
    st.subheader("Abstractive Summary"); st.write(abstract_summary)

    all_docs = store.get(include=["documents"])["documents"]
    trend = compute_sentiment_trend(all_docs)
    st.subheader("Sentiment Trend")
    fig = plot_sentiment_trend(trend); st.pyplot(fig)

    pdf_bytes = export_pdf(summary_json, abstract_summary, sentiment, fig)
    st.download_button("📥 Download PDF Report", data=pdf_bytes,
                       file_name=f"{company_name}_AI_Report.pdf", mime="application/pdf")
else:
    st.info("👆 Upload a financial report to begin dynamic analysis.")
