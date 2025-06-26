import os
import streamlit as st
from PIL import Image
from ragchain_langchain import qa_chain
import app_evaluate_ui  # your evaluation UI function

# Set Streamlit page config
st.set_page_config(page_title="🌾 Rice Farming Agent", layout="wide")

# Read query param for page routing
page = st.query_params.get("page", "chat")

# === PAGE: Main Chat UI ===
if page == "chat":
    # Banner and title
    col1, col2 = st.columns([0.80, 0.15])
    with col2:
        st.markdown(
            """
            <a href="?page=evaluation" target="_self" style="text-decoration: none;">
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
    <span style="font-size: 16px;">📊</span>
    <span>Evaluation Metrics</span>
  </button>
</a>

            """, unsafe_allow_html=True
        )
    img_path = os.path.join(os.path.dirname(__file__), "Rice Farming.png")
    img = Image.open(img_path)
    st.image(img.resize((700, 400)))
    st.markdown("## 🌾 Rice Farming Assistance Agent")

    # Top-right: Navigation to Evaluation UI
    st.divider()

    # Init session state for chat log
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # Show chat history
    for entry in st.session_state.chat_log:
        role = entry["role"]
        avatar = "🧑‍🌾" if role == "user" else "🤖🚜"
        st.markdown(f"**{avatar} {role.capitalize()}:**\n\n{entry['content']}")

    # Chat input box
    query = st.chat_input("Ask about rice farming...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.chat_log.append({"role": "user", "content": query})

        result = qa_chain.invoke({"question": query})
        answer = result["answer"]
        docs = result.get("source_documents", [])

        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_log.append({"role": "assistant", "content": answer})

        # Sources and chunks
        if docs:
            with st.expander("📚 Sources used"):
                shown = set()
                for doc in docs:
                    meta = doc.metadata
                    title = meta.get("title", "No title")
                    url = meta.get("url", "N/A")
                    if (title, url) not in shown:
                        shown.add((title, url))
                        st.markdown(f"- **{title}**\n  🔗[Open Source]({url})")

            with st.expander("📄 Chunks retrieved"):
                for doc in docs:
                    meta = doc.metadata
                    title = meta.get("title", "No title")
                    score = meta.get("score", "?")
                    chunk_id_full = meta.get("chunk_id", "N/A")
                    chunk_id = chunk_id_full.split("_")[-1] if "chunk" in chunk_id_full else chunk_id_full

                    st.markdown(
                        f"### 🧩 {title}"
                        f"\n- **Chunk:** `{chunk_id}` | **Score:** `{score:.3f}`\n\n"
                        f"_Summary: {meta.get('summary', '')}_\n\n"
                        f"**Preview:** {doc.page_content[:200]}..."
                    )

# === PAGE: Evaluation Metrics ===
elif page == "evaluation":
    app_evaluate_ui.render_evaluation_ui()  # must be a function in your app_evaluate_ui.py
