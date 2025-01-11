# RAG-Based Chatbot with LangChain and Hugging Face

This project implements a Retrieval-Augmented Generation (RAG)-based chatbot using open-source tools like LangChain and Hugging Face models. The chatbot leverages a dataset as its knowledge base to provide accurate, context-aware answers to user queries.

---

## Features
- Uses **Hugging Face embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) for document vectorization.
- Employs Hugging Face's **GPT-2** for text generation.
- Implements LangChain's Retrieval-Augmented Generation pipeline.
- Interactive chatbot interface built with **Streamlit**.
- No paid APIs or secret keys required.

---

## File Structure
```
rag_chatbot_project/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── data/
│   └── data.csv          # Dataset used as the knowledge base
├── utils/
│   └── rag_utils.py      # Utility functions for loading data and setting up RAG pipeline
└── README.md             # Documentation for the project
```

---

## Setup Instructions

### 1. Create a Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
- Add your dataset file (CSV format) to the `data/` folder.
- Ensure the file is named `data.csv` or update the path in `app.py`.

### 4. Run the Application Locally
```bash
streamlit run app.py
```

---

## Example Dataset Format
```csv
Question,Answer
"What is LangChain?","LangChain is a framework for building applications with LLMs."
"What is GPT-2?","GPT-2 is a generative language model developed by OpenAI."
```

---

## Online Deployment
The chatbot is deployed on Streamlit Community Cloud. You can access it here: [AI Bot Chat](https://aibotchat.streamlit.app/)

---

## Additional Notes
- The project can be extended to use other open-source language models like **GPT-4All** or **LLaMA**.
- For large datasets, consider using a vector database like **Pinecone** or **Weaviate**.

---

## Contribution
Feel free to fork this repository, raise issues, or submit pull requests to improve the chatbot.

---

## License
This project is licensed under the MIT License.
