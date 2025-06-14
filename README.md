# 📚 Semantic Book Recommender

Welcome to the **Semantic Book Recommender** – an intelligent system that recommends books based on natural language queries, genre, and emotional tone using state-of-the-art semantic search and emotion classification.

![UI Screenshot](https://github.com/user-attachments/assets/85b2e655-b7d7-4825-a1f1-932d6c2bceee)

---

## 🚀 Features

### ✅ Natural Language Recommendations
- Enter a sentence like *"A story about forgiveness and second chances"* and get context-aware book suggestions.
- Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity search.

### 🧠 Emotion-Aware Filtering
- Recommend books based on tone:  
  **Happy**, **Sad**, **Angry**, **Surprising**, or **Suspenseful**.
- Emotion scores are precomputed and sorted accordingly.

### 📂 Genre-Based Filtering
- Filter by book category (e.g., *Fiction*, *Science*, *Biography*).
- Easily combine genre + tone + query.

### 🖼️ Beautiful Gallery Interface
- Built with [Gradio](https://gradio.app/) and styled using **Glassmorphism**.
- Book cover, title, author, and short description are shown in an elegant card grid.

### 🧠 Vector Database Powered by LangChain + Chroma
- `tagged_description.txt` is embedded and chunked to enable fast semantic matching.
- Uses `LangChain`, `Chroma`, and `HuggingFaceEmbeddings`.

---

## 📦 Tech Stack

| Feature        | Stack / Library                                |
|----------------|-------------------------------------------------|
| UI             | Gradio (`gr.Blocks`, `Gallery`, `Dropdown`)     |
| ML Embeddings  | HuggingFace `sentence-transformers`             |
| Vector Store   | `Chroma` via `LangChain`                        |
| Data Handling  | `Pandas`, `NumPy`                               |
| Deployment     | Localhost or shareable Gradio link              |

---

## ⚙️ How It Works

1. **User Query** → User enters a sentence (e.g., “A tale of courage and loss”).
2. **Semantic Search** → Text is embedded and compared to the `tagged_description.txt` using `MiniLM`.
3. **Book Retrieval** → Top similar books (by ISBN) are fetched from `books_with_emotions.csv`.
4. **Filtering**:
   - By **Category** (if selected)
   - By **Tone** (emotion-sorted)
5. **Display** → Recommended books shown with title, authors, image, and description.

---

## 🛠️ Installation & Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Book-Recommender.git
cd Book-Recommender

# 2. Setup environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python gradio-dashboard.py
