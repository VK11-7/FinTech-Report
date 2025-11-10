import os, json, io, fitz, matplotlib.pyplot as plt
import streamlit as st
from typing import List, Dict, Any
from scipy.special import softmax
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from fpdf import FPDF

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------------- CONFIG ----------------
FINBERT_MODEL = "ProsusAI/finbert"
INSTRUCTION_MODEL = "google/flan-t5-large"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_fin"
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

def build_vectorstore(docs: List[str], embedding_fn):
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_fn)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = []
    for d in docs:
        chunks.extend(splitter.split_documents([Document(page_content=d)]))
    return Chroma.from_documents(chunks, embedding=embedding_fn, persist_directory=CHROMA_DIR)

def retrieve_context(store: Chroma, query: str, k=3) -> str:
    retriever = store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)
    return "\n\n".join([r.page_content for r in results])

def generate_structured_summary(text: str, context: str, sentiment: Dict[str, float]) -> Dict[str, Any]:
    text, context = text[:1500], context[:1000]
    prompt = f"""
Summarize the following REPORT into structured JSON.

Schema:
{{
  "summary": "One-sentence summary.",
  "drivers": ["2-3 key business drivers"],
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
Write a fluent 2–3 sentence summary using CONTEXT and FINBERT sentiment.

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

def export_to_pdf(summary_json: dict, abstract_summary: str, sentiment: dict, fig) -> bytes:
    """Export all analysis results to a single downloadable PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Financial Report Analysis", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"\nOverall Sentiment:\n{json.dumps(sentiment, indent=2)}")
    pdf.multi_cell(0, 10, f"\nStructured Summary:\n{json.dumps(summary_json, indent=2)}")
    pdf.multi_cell(0, 10, f"\nAbstractive Summary:\n{abstract_summary}")

    # Save trend plot to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png"); buf.seek(0)
    img_path = "trend_temp.png"
    with open(img_path, "wb") as f:
        f.write(buf.read())
    pdf.image(img_path, x=10, w=180)
    os.remove(img_path)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Financial Report Analyzer", layout="wide")
st.title("💼 Generative AI for Financial Report Summarization with PDF Export")

uploaded_file = st.sidebar.file_uploader("📂 Upload Financial Report (.pdf or .txt)", type=["pdf", "txt"])
historical_docs = [
    "Q1 2025: Stable revenue but margins declined due to raw-material inflation.",
    "Q2 2025: Strong demand recovery and improved cost control.",
    "Q3 2025: Operating margin up 10% from higher production efficiency."
]
store = build_vectorstore(historical_docs, embedding_fn)

if uploaded_file:
    report_text = extract_text(uploaded_file)
    st.subheader("1️⃣ Report Preview")
    st.text_area("Extracted Report", report_text[:2000] + "...", height=200)

    st.subheader("2️⃣ FinBERT Sentiment")
    sentiment = finbert_sentiment(report_text)
    st.json(sentiment)

    st.subheader("3️⃣ RAG Context")
    context = retrieve_context(store, report_text)
    st.text_area("Retrieved Context", context, height=150)

    st.subheader("4️⃣ Structured JSON Summary")
    summary_json = generate_structured_summary(report_text, context, sentiment)
    st.json(summary_json)

    st.subheader("5️⃣ Abstractive Narrative Summary")
    abstract_summary = generate_abstractive_summary(report_text, context, sentiment)
    st.write(abstract_summary)

    st.subheader("6️⃣ Sentiment Trend Analysis")
    all_texts = historical_docs + [report_text]
    trend = compute_sentiment_trend(all_texts)
    fig = plot_sentiment_trend(trend)
    st.pyplot(fig)

    pdf_data = export_to_pdf(summary_json, abstract_summary, sentiment, fig)
    st.download_button(
        label="📥 Download Full Report (PDF)",
        data=pdf_data,
        file_name="Financial_AI_Report.pdf",
        mime="application/pdf"
    )
    st.success("✅ Analysis complete!")
else:
    st.info("👆 Upload a financial report to start analysis.")