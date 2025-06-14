import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import gradio as gr

load_dotenv()

# Load the book dataset
books = pd.read_csv("books_with_emotions.csv")

# Process thumbnails
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(), "cover-not-found.png", books["large_thumbnail"]
)

# Create LangChain Document objects with metadata (isbn13)
documents = [
    Document(page_content=row["description"], metadata={"isbn13": str(row["isbn13"])})
    for _, row in books.iterrows()
    if pd.notna(row["description"]) and row["description"].strip() != ""
]

# Split and embed
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(split_docs, embedding=embedding)

def retrieve_semantics_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract isbn13 from metadata
    matched_isbns = [rec.metadata["isbn13"] for rec in recs]
    books_recs = books[books["isbn13"].astype(str).isin(matched_isbns)]

    # Filter by category
    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category]

    # Tone filtering
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        sort_col = tone_map[tone]
        books_recs = books_recs.sort_values(by=sort_col, ascending=False)

    return books_recs.head(final_top_k)

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantics_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row.get("description", "")
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# Dropdown options
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select a tone", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## 🎯 Recommendations")

    output = gr.Gallery(label="Recommended Books", columns=4, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

# Run the Gradio app
if __name__ == "__main__":
    dashboard.launch()
