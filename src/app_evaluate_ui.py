import streamlit as st
import pandas as pd
import json
from io import BytesIO

from generate_groundtruth_app import run_groundtruth_generation
from evaluate_app import run_ragas_evaluation_from_data
from evaluate_rag_metrics_app import run_rag_evaluator_from_data

def render_evaluation_ui():
    st.set_page_config(page_title="RAG Evaluation", layout="wide")
    st.title("📊 RAG Evaluation Dashboard")

    # Top-right: Back to main chat
    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        st.markdown(
            """
            <a href="?page=chat" target="_self" style="text-decoration: none;">
            <button style="
              background-color: #f8f9fa;
              border: 1px solid #ccc;
              padding: 8px 16px;
              font-size: 14px;
              border-radius: 6px;
              color: #333;
              display: flex;
              align-items: center;
              gap: 8px;
              cursor: pointer;
              box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
              transition: background-color 0.2s ease;
          " onmouseover="this.style.backgroundColor='#e2e6ea'" onmouseout="this.style.backgroundColor='#f8f9fa'">
            <span style="font-size: 16px;">⬅️</span>
            <span>Back to Chat</span>
          </button>
        </a>
            """, unsafe_allow_html=True
        )


    # ---------- Upload & Groundtruth Generation ---------- #
    st.subheader("1. Upload PDFs and Generate Groundtruth")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    n_questions = st.number_input(
        label="Choose how many QA pairs to generate per PDF",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="⚠️ Higher values increase cost and latency (more tokens used per API call)."
    )

    if uploaded_files and st.button("⚙️ Generate Groundtruth JSON"):
        with st.spinner("Generating questions from uploaded PDFs..."):
            qa_data, json_blob = run_groundtruth_generation(uploaded_files, questions_per_pdf=n_questions)

        if qa_data and json_blob:
            st.session_state.qa_data = qa_data
            st.session_state.json_blob = json_blob
            st.success("✅ Groundtruth generation successful!")

            st.download_button(
                "⬇️ Download Groundtruth JSON",
                data=json_blob,
                file_name="groundtruth.json",
                mime="application/json"
            )
        else:
            st.warning("⚠️ No valid QA pairs were generated.")

    # ---------- Evaluation Buttons ---------- #
    st.subheader("2. Select Evaluation Method(s)")
    eval_col1, eval_col2 = st.columns(2)
    run_ragas = eval_col1.button("🚀 Run RAGAS Evaluation")
    run_rag_eval = eval_col2.button("📐 Run RAG Evaluator Evaluation")

    # ---------- RAGAS Evaluation ---------- #
    if run_ragas:
        if "qa_data" not in st.session_state:
            st.error("❌ Please generate groundtruth questions first.")
        else:
            with st.spinner("Running RAGAS Evaluation..."):
                df, csv_blob, err = run_ragas_evaluation_from_data(st.session_state.qa_data)

            if err:
                st.error(f"❌ {err}")
            elif df is not None:
                st.success("✅ RAGAS Evaluation Completed")
                st.session_state.ragas_scores = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().round(3)
                st.download_button(
                    "⬇️ Download RAGAS Score CSV",
                    data=csv_blob,
                    file_name="ragas_scores.csv",
                    mime="text/csv"
                )

    if "ragas_scores" in st.session_state:
        st.markdown("#### 📈 RAGAS Scores")
        scores = st.session_state.ragas_scores
        col1, col2 = st.columns(2)
        col1.metric("Faithfulness", scores["faithfulness"])
        col1.metric("Answer Relevancy", scores["answer_relevancy"])
        col2.metric("Context Precision", scores["context_precision"])
        col2.metric("Context Recall", scores["context_recall"])

    # ---------- RAG Evaluator ---------- #
    if run_rag_eval:
        if "qa_data" not in st.session_state:
            st.error("❌ Groundtruth must be generated first.")
        else:
            with st.spinner("Running RAG Evaluator..."):
                df_eval, csv_blob_eval, err = run_rag_evaluator_from_data(st.session_state.qa_data)

            if err:
                st.error(f"❌ {err}")
            elif df_eval is not None:
                st.success("✅ RAG Evaluator Completed")
                st.session_state.show_rag_eval_explainer = True
                st.download_button(
                    "⬇️ Download Evaluator Metrics CSV",
                    data=csv_blob_eval,
                    file_name="rag_evaluator_scores.csv",
                    mime="text/csv"
                )

    if st.session_state.get("show_rag_eval_explainer"):
        st.markdown("""
        ### 📊 RAG Evaluator Metrics (Explained)

        ---

        **✳️ BLEU (0–100)**
        *Measures n-gram overlap between output and reference.*
        &nbsp;&nbsp;&nbsp;&nbsp;• 0–20: Low
        &nbsp;&nbsp;&nbsp;&nbsp;• 20–40: Medium-low
        &nbsp;&nbsp;&nbsp;&nbsp;• 40–60: Medium
        &nbsp;&nbsp;&nbsp;&nbsp;• 60–80: High
        &nbsp;&nbsp;&nbsp;&nbsp;• 80–100: Very high

        **✳️ ROUGE-1 (0–1)**
        *Measures unigram recall between output and reference.*
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.0–0.2: Poor
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.2–0.4: Fair
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.4–0.6: Good
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.6–0.8: Very good
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.8–1.0: Excellent

        **✳️ BERT Score (0–1)**
        *Evaluates semantic similarity using embeddings.*
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.0–0.5: Low
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.5–0.7: Moderate
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.7–0.8: Good
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.8–0.9: High
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.9–1.0: Very high

        **✳️ Perplexity (1–∞)**
        *Indicates language model confidence (lower is better).*
        &nbsp;&nbsp;&nbsp;&nbsp;• 1–10: Excellent
        &nbsp;&nbsp;&nbsp;&nbsp;• 10–50: Good
        &nbsp;&nbsp;&nbsp;&nbsp;• 50–100: Moderate
        &nbsp;&nbsp;&nbsp;&nbsp;• 100+: High / possibly incoherent

        **✳️ Diversity (0–1)**
        *Measures bigram uniqueness in the response.*
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.0–0.2: Very low
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.2–0.4: Low
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.4–0.6: Moderate
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.6–0.8: High
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.8–1.0: Very high

        **✳️ Racial Bias (0–1)**
        *Estimates presence of biased or harmful language.*
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.0–0.2: Low
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.2–0.4: Moderate
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.4–0.6: High
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.6–0.8: Very high
        &nbsp;&nbsp;&nbsp;&nbsp;• 0.8–1.0: Extreme
        """)




# import streamlit as st
# import pandas as pd
# import json
#
# from generate_groundtruth_app import run_groundtruth_generation
# from evaluate_app import run_ragas_evaluation_from_data
# from evaluate_rag_metrics_app import run_rag_evaluator_from_data
#
# st.set_page_config(page_title="RAG Evaluation", layout="wide")
#
# st.title("📊 RAG Evaluation Dashboard")
#
# # ---------- Refresh Button ---------- #
# col1, col2 = st.columns([0.85, 0.15])
# with col2:
#     if st.button("🔄 Refresh App"):
#         st.session_state.clear()
#         st.rerun()
#
# # ---------- Upload & Groundtruth Generation ---------- #
# st.subheader("1. Upload PDFs and Generate Groundtruth")
# uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
#
# n_questions = st.number_input(
#     label="Choose how many QA pairs to generate per PDF",
#     min_value=1,
#     max_value=15,
#     value=5,
#     step=1,
#     help="⚠️ Higher values increase cost and latency (more tokens used per API call)."
# )
#
# if uploaded_files and st.button("⚙️ Generate Groundtruth JSON"):
#     with st.spinner("Generating questions from uploaded PDFs..."):
#         qa_data, json_blob = run_groundtruth_generation(uploaded_files, questions_per_pdf=n_questions)
#
#     if qa_data and json_blob:
#         st.session_state.qa_data = qa_data
#         st.session_state.json_blob = json_blob
#         st.success("✅ Groundtruth generation successful!")
#
#         st.download_button(
#             "⬇️ Download Groundtruth JSON",
#             data=json_blob,
#             file_name="groundtruth.json",
#             mime="application/json"
#         )
#     else:
#         st.warning("⚠️ No valid QA pairs were generated.")
#
# # ---------- Evaluation Section ---------- #
# st.subheader("2. Select Evaluation Method(s)")
# eval_col1, eval_col2 = st.columns(2)
# run_ragas = eval_col1.button("🚀 Run RAGAS Evaluation")
# run_rag_eval = eval_col2.button("📐 Run RAG Evaluator Evaluation")
#
# # ---------- RAGAS Evaluation ---------- #
# if run_ragas:
#     if "qa_data" not in st.session_state:
#         st.error("❌ Please generate groundtruth questions first.")
#     else:
#         with st.spinner("Running RAGAS Evaluation..."):
#             df, csv_blob, err = run_ragas_evaluation_from_data(st.session_state.qa_data)
#
#         if err:
#             st.error(f"❌ {err}")
#         elif df is not None:
#             st.session_state.ragas_scores = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().round(3)
#             st.session_state.ragas_csv_blob = csv_blob
#             st.success("✅ RAGAS Evaluation Completed")
#
#             st.download_button(
#                 "⬇️ Download RAGAS Score CSV",
#                 data=csv_blob,
#                 file_name="ragas_scores.csv",
#                 mime="text/csv"
#             )
#
# if "ragas_scores" in st.session_state:
#     st.markdown("#### 📈 RAGAS Scores")
#     scores = st.session_state.ragas_scores
#     col1, col2 = st.columns(2)
#     col1.metric("Faithfulness", scores["faithfulness"])
#     col1.metric("Answer Relevancy", scores["answer_relevancy"])
#     col2.metric("Context Precision", scores["context_precision"])
#     col2.metric("Context Recall", scores["context_recall"])
#
# # ---------- RAG Evaluator ---------- #
# if run_rag_eval:
#     if "qa_data" not in st.session_state:
#         st.error("❌ Groundtruth must be generated first.")
#     else:
#         with st.spinner("Running RAG Evaluator..."):
#             df_eval, csv_blob_eval, err = run_rag_evaluator_from_data(st.session_state.qa_data)
#
#         if err:
#             st.error(f"❌ {err}")
#         elif df_eval is not None:
#             st.success("✅ RAG Evaluator Completed")
#             st.session_state.show_rag_eval_explainer = True
#
#             st.download_button(
#                 "⬇️ Download Evaluator Metrics CSV",
#                 data=csv_blob_eval,
#                 file_name="rag_evaluator_scores.csv",
#                 mime="text/csv"
#             )
#
# if st.session_state.get("show_rag_eval_explainer"):
#     st.markdown("""
#     ### 📊 RAG Evaluator Metrics (Explained)
#
#     ---
#
#     **✳️ BLEU (0–100)**
#     *Measures n-gram overlap between output and reference.*
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0–20: Low
#     &nbsp;&nbsp;&nbsp;&nbsp;• 20–40: Medium-low
#     &nbsp;&nbsp;&nbsp;&nbsp;• 40–60: Medium
#     &nbsp;&nbsp;&nbsp;&nbsp;• 60–80: High
#     &nbsp;&nbsp;&nbsp;&nbsp;• 80–100: Very high
#
#     **✳️ ROUGE-1 (0–1)**
#     *Measures unigram recall between output and reference.*
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.0–0.2: Poor
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.2–0.4: Fair
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.4–0.6: Good
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.6–0.8: Very good
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.8–1.0: Excellent
#
#     **✳️ BERT Score (0–1)**
#     *Evaluates semantic similarity using embeddings.*
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.0–0.5: Low
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.5–0.7: Moderate
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.7–0.8: Good
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.8–0.9: High
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.9–1.0: Very high
#
#     **✳️ Perplexity (1–∞)**
#     *Indicates language model confidence (lower is better).*
#     &nbsp;&nbsp;&nbsp;&nbsp;• 1–10: Excellent
#     &nbsp;&nbsp;&nbsp;&nbsp;• 10–50: Good
#     &nbsp;&nbsp;&nbsp;&nbsp;• 50–100: Moderate
#     &nbsp;&nbsp;&nbsp;&nbsp;• 100+: High / possibly incoherent
#
#     **✳️ Diversity (0–1)**
#     *Measures bigram uniqueness in the response.*
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.0–0.2: Very low
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.2–0.4: Low
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.4–0.6: Moderate
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.6–0.8: High
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.8–1.0: Very high
#
#     **✳️ Racial Bias (0–1)**
#     *Estimates presence of biased or harmful language.*
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.0–0.2: Low
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.2–0.4: Moderate
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.4–0.6: High
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.6–0.8: Very high
#     &nbsp;&nbsp;&nbsp;&nbsp;• 0.8–1.0: Extreme
#     """)
#
#
#
