# 📊 Generative AI for Financial Report Summarization

An AI-powered system that generates context-aware summaries of financial reports using a Retrieval-Augmented Generation (RAG) pipeline. The system combines domain-specific embeddings with transformer-based text generation to provide accurate and structured summaries for decision support.

---

## 📌 Overview

This project focuses on automating the summarization of complex financial documents by integrating:

- Domain-specific embeddings (FinBERT)
- Contextual document retrieval (ChromaDB)
- Transformer-based summarization (FLAN-T5)

The system enhances financial analysis by producing **relevant, concise, and context-aware summaries**, enabling faster and more informed decision-making.

---

## 🧠 Key Features

- 📄 **Financial Document Processing**
  - Extracts text from PDF reports using PyMuPDF  
  - Handles large and unstructured financial documents  

- 🔍 **Contextual Retrieval (RAG Pipeline)**
  - Uses **ChromaDB** for vector storage and retrieval  
  - Retrieves the most relevant sections of documents for summarization  

- 🧬 **Domain-Specific Embeddings**
  - Utilizes **FinBERT** for financial text understanding  
  - Improves relevance in financial context compared to generic embeddings  

- 🤖 **Structured Summarization**
  - Uses **FLAN-T5** for generating coherent summaries  
  - Produces structured and meaningful outputs  

- 🌐 **Interactive Streamlit Interface**
  - Upload financial reports  
  - Generate summaries in real-time  
  - Easy-to-use UI for analysis  

---

## 🏗️ System Architecture

Financial Report (PDF) <br>
│ <br>
▼ <br>
Text Extraction (PyMuPDF) <br>
│ <br>
▼ <br>
Text Chunking & Cleaning <br>
│ <br>
▼ <br>
Embedding Generation (FinBERT) <br>
│ <br>
▼ <br>
Vector Storage (ChromaDB) <br>
│ <br>
▼ <br>
Context Retrieval (Top-K) <br>
│ <br>
▼ <br>
Summarization (FLAN-T5) <br>
│ <br>
▼ <br>
Streamlit UI Output <br>


---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit  
- **Embeddings:** FinBERT, SentenceTransformers  
- **LLM / Summarization:** FLAN-T5  
- **Vector Database:** ChromaDB  
- **PDF Processing:** PyMuPDF  
- **Data Processing:** NumPy  

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/financial-rag-summarizer.git
cd financial-rag-summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

📈 Evaluation Metrics <br>
Relevance of retrieved context <br>
Summary coherence and fluency <br>
Information coverage <br> 
Reduction in document length <br>

---

Varadharajan K
