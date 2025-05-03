import streamlit as st
from rag_agent import create_agent
import re

st.set_page_config(page_title="TCS Buddy", layout="wide")

st.title("🤖 TCS Buddy – Employee Support AI Agent")

query = st.text_input("Ask a question about TCS processes...")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = create_agent()


if query:
    with st.spinner("TCS Buddy is thinking..."):
        output = st.session_state.qa_chain.invoke({"query": query})
        result = output["result"]

        # Try to extract one-line answer from structured "Answer (...): ..." pattern
        match = re.search(r"Answer\s*(?:\(.*?\))?:\s*(.*)", result, re.IGNORECASE)
        one_liner = match.group(1).strip() if match else result.strip()

        # Display only the final answer
        st.subheader("Answer:")
        st.markdown(one_liner)