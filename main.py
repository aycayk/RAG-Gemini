import streamlit as st
from conversation_utils import (get_recent_qa_pairs, compress_conversation_context, expand_user_question, create_final_prompt)
from pdf_utils import extract_pdf_files
from model_utils import build_combined_index, retrieve_relevant_chunks_combined, query_gemini
from chat_prompt import create_prompt

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
api_key = st.secrets["API_KEY"]

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    model_info = {
        "all-MiniLM-L6-v2": "Fast & efficient, general semantic search",
        "all-mpnet-base-v2": "Higher accuracy, detailed analysis",
        "paraphrase-MiniLM-L6-v2": "Paraphrase analysis",
        "distiluse-base-multilingual-cased-v2": "Multilingual support",
        "all-distilroberta-v1": "Alternative, robust model",
        "allenai-specter": "Scientific articles, technical and mathematical content",
        # "mathbert": "Optimized for mathematical expressions"
    }

    st.sidebar.header("PDF Processing")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs or ZIP", type=["pdf", "zip"], accept_multiple_files=True)
    model_name = st.sidebar.selectbox(
        "Model Selection",
        options=list(model_info.keys()),
        format_func=lambda x: f"{x} - {model_info[x]}",
        help="Select the model to analyze the article content"
    )
    chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=800, value=500, step=50)
    if uploaded_files and st.sidebar.button("Process Articles"):
        with st.sidebar:
            pdf_files = extract_pdf_files(uploaded_files)
            model_emb, combined_index, all_metadata = build_combined_index(pdf_files, model_name=model_name, chunk_size=chunk_size)
            st.session_state.combined_index = combined_index
            st.session_state.model_emb = model_emb
            st.session_state.all_metadata = all_metadata
            st.success("All articles processed successfully!")
            st.session_state.chat_history = []

    st.title("RAGemini Scholar")
    st.markdown("<p class='header-title'>Using Google's Gemini 2.0 Flash model, this project processes and analyzes uploaded articles to create a dynamic platform for accurate Q&A interactions.</p>", unsafe_allow_html=True)

    if "combined_index" not in st.session_state:
        st.warning("Please upload and process PDFs first.")
        return

    st.subheader("Conversation History")
    for msg in st.session_state.chat_history:
        st.markdown(f"""
        <div class='chat-msg user'><strong>User:</strong> {msg['question']}</div>
        <div class='chat-msg bot'><strong>Bot:</strong> {msg['answer']}</div>
        """, unsafe_allow_html=True)

    with st.form("query_form"):
        query = st.text_input("Enter your query:")
        submitted = st.form_submit_button("Get Answer")
    
    if submitted and query:
        # Eğer chat geçmişi boşsa (ilk soru) compress/expand adımlarını atla.
        if not st.session_state.chat_history:
            # İlk sorgu için doğrudan orijinal query üzerinden PDF'lerden bilgi çek
            retrieved_results = retrieve_relevant_chunks_combined(
                query, 
                st.session_state.model_emb, 
                st.session_state.combined_index, 
                st.session_state.all_metadata, 
                top_k=3
            )
            # Direkt create_prompt kullanarak prompt oluştur (chat_history boş)
            final_prompt = create_prompt(query, retrieved_results, [])
            with st.expander("Generated Prompt"):
                st.code(final_prompt, language="text")
            result = query_gemini(final_prompt, api_key)
            answer = ""
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    answer += part.get("text", "")
            st.session_state.chat_history.append({"question": query, "answer": answer})
        else:
            # Takip soruları için: önce geçmiş Q/A çiftlerini al, compress ve expand işlemlerini uygula.
            qa_pairs = get_recent_qa_pairs(st.session_state.chat_history)
            compressed_context = compress_conversation_context(qa_pairs, api_key)
            expanded_question = expand_user_question(query, compressed_context, api_key)
            retrieved_results = retrieve_relevant_chunks_combined(
                expanded_question, 
                st.session_state.model_emb, 
                st.session_state.combined_index, 
                st.session_state.all_metadata, 
                top_k=3
            )
            final_prompt = create_final_prompt(compressed_context, expanded_question, retrieved_results)
            with st.expander("Generated Prompt"):
                st.code(final_prompt, language="text")
            result = query_gemini(final_prompt, api_key)
            answer = ""
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    answer += part.get("text", "")
            st.session_state.chat_history.append({"question": query, "answer": answer})
        
        st.subheader("Answer")
        st.write(answer)
    
    st.markdown("""
    <style>
    .header-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 500;
        color: #555555;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .chat-msg {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .user {
        background-color: #333333;
        color: white;
    }
    .bot {
        background-color: #00008B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
