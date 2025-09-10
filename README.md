# Receipt Text Extractor (Ollama + Vision Model)

This Python script extracts structured text data from receipt images using [Ollama](https://ollama.com/) and a multimodal LLM (default: `llama3.2-vision`).  
It processes the image, sends it to the model, and returns **strictly valid JSON** with merchant, items, date, and totals.

---

## Features

- **Image preprocessing**: resizes large images to a max side of 1280px for efficiency.  
- **Structured JSON output**: always follows the same schema (merchant, items, totals, VAT, etc.).  
- **Strict format**: model is instructed to return *only JSON* (no explanations or extra text).  
- **CLI usage**: specify receipt image path and optional model.  
- **Automatic cleanup**: temporary image files are deleted after extraction.  

---
