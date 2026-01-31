import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl

st.set_page_config(
    page_icon="📚",   # emoji as logo
    page_title="My Streamlit App",
    layout="wide"
)

st.title("📚 Book Recommendor")

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0e1117;
    color: #ffffff;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>

<div class="footer">
    © 2026 | Built by Deepesh Patel | Streamlit App
</div>
""", unsafe_allow_html=True)

bookTitle=pkl.load(open('bookTitle.pkl','rb'))
bookPoster=pkl.load(open('posterUrl.pkl','rb'))
with open("model.pkl", "rb") as f:
    tfidf, vectors = pkl.load(f)

def recommend(query, n=5):
    if not isinstance(query, str) or query.strip() == "":
        return "Please provide a valid input"

    # 1. Vectorize the query
    query_vector = tfidf.transform([query])

    # 2. Compute similarity (safe: 1 x N)
    sim_scores = cosine_similarity(query_vector, vectors).flatten()

    # 3. Get top N results
    top_indices = np.argsort(sim_scores)[::-1][1:n+1]

    # 4. Return recommendations
    return top_indices

# st.logo("WhatsApp Image 2026-01-07 at 20.59.21.jpeg",size="large")
st.header("Welcome to the Book Recommender")



recommenderType=st.selectbox("Select the Recommender type......",['Based on Book Name','Based on Keywords'])

if recommenderType=='Based on Book Name':  
    query=st.selectbox("Enter the Book......",bookTitle)

elif recommenderType=='Based on Keywords':
    query=st.text_input("Enter Book related Keywords !",icon="🔥",placeholder="Enter title / author / year / publisher")

if st.button("Recommend"):
    results=recommend(query)
    col1,col2,col3,col4,col5= st.columns(5)

    with col1:
        st.image(bookPoster[results[0]],width=180,)
        st.write(bookTitle[results[0]])
    with col2:
        st.image(bookPoster[results[1]],width=180,)
        st.write(bookTitle[results[1]])
    with col3:
        st.image(bookPoster[results[2]],width=180,)
        st.write(bookTitle[results[2]])
    with col4:
        st.image(bookPoster[results[3]],width=180,)
        st.write(bookTitle[results[3]])
    with col5:
        st.image(bookPoster[results[4]],width=180,)
        st.write(bookTitle[results[4]])


