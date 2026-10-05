<div align="center">
  <a href="https://youtu.be/bNQp-Qf5neo">
    <img src="https://img.youtube.com/vi/bNQp-Qf5neo/0.jpg" alt="TeleOCR: The 1B OCR Model Challenging GPT-5.2 and Gemini!">
  </a>
  <h3>📺 <a href="https://youtu.be/bNQp-Qf5neo">Watch the full tutorial on YouTube</a></h3>
</div>

# 🌐 TeleOCR 1B Beats GPT

TeleOCR is a lightweight 1.2B parameter Vision-Language Model that replaces commercial cloud Document AI services. It parses camera-captured receipts, contracts, and math tables directly into structured JSON and SQLite databases offline.

---

## ⚡ Quick 3-Step Workflow

| Step 1: Input | Step 2: AI Action | Step 3: Result |
| :--- | :--- | :--- |
| 📷 Photographed Invoice / Contract | 🧠 TeleOCR Geometry & OTSL Grid Parsing | 📊 Structured JSON & SQLite Database |

---

## 🚀 Key Features

- **🏆 OmniDocBench #1 Leader:** Achieves **96.87** overall score, outperforming GPT-5.2 (86.52) and Gemini 3 Pro (92.85).
- **📷 Dewarping-Free Camera OCR:** Parses angled, crumpled, or unevenly lit phone photos of documents without external dewarping tools.
- **📊 Table Grid Decoupling:** Converts complex nested data tables directly into HTML grid structures using One-Table-Structure-Language (OTSL).
- **🔒 100% Offline Autonomy:** All document extraction runs locally on your machine with zero recurring cloud API fees or privacy risks.

---

## 📦 File Structure

```
TeleOCR/
├── main.py
├── requirements.txt
├── README.md
└── tests/
    └── test_main.py
```

---

## 🛠️ Installation & TeleOCR Model Setup

Copy and paste the following single PowerShell command to install the TeleOCR dependencies:

```powershell
pip install transformers torch pillow
```

To download the TeleOCR model weights directly from Hugging Face:

```powershell
hf download StarDoc-AI/TeleOCR --local-dir models/TeleOCR
```

---

## 💻 How to Run

Run the automated document parsing loop across all input files:

```powershell
python main.py
```

The script automatically scans all document images in `inputs/`, runs the TeleOCR parsing loop, saves records to SQLite (`outputs/documents.db`), and builds the final Markdown report (`outputs/outputs.md`).

---

## 🎯 5 Practical Real-World Use Cases

1. **Receipt & Expense Automation:** Convert phone photos of paper receipts into structured JSON for accounting software.
2. **Contract Clause Extraction:** Extract multi-column legal terms and nested billing schedules into local databases.
3. **STEM Academic Digitization:** Parse research papers to convert mathematical equations into LaTeX code blocks.
4. **Supply Chain Invoice Processing:** Parse messy shipping manifests and warehouse tables into structured database entries.
5. **Private On-Premise Archiving:** Process sensitive financial and medical documents without uploading data to public cloud APIs.

---

## 🔮 5 Future Enhancement Roadmap Items

1. **GUI Dashboard Interface:** Interactive desktop dashboard for drag-and-drop document upload and batch JSON preview.
2. **Multi-Language OCR Expansion:** Native support for non-Latin scripts and low-resource historical manuscripts.
3. **Automated Vector Embedding Pipeline:** Direct integration with local vector databases for RAG document search.
4. **Mobile Camera App Integration:** On-device camera scanning powered by quantized GGUF models.
5. **Custom Schema Mapper:** User-defined JSON schema mapping for enterprise ERP software integrations.

---

## 🏷️ Keywords & SEO

`TeleOCR` `Vision-Language Model` `Document AI` `OTSL` `Offline OCR` `Receipt Parsing` `Hugging Face Models` `Local AI Model` `Python Document Processing` `SQLite Export`
