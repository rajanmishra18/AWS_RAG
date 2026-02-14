import os
import streamlit as st 
import requests

st.set_page_config(page_title="RAG Assistant", page_icon="🧠")

st.title("🧠 RAG Assistant")
st.caption('As questions grounded in your document')

API_URL= "http://16.170.254.194:8000/"

question=st.text_input("Ask your question")

if st.button('Ask'):
    if not question.strip():
        st.warning('Please enter a question')
    else:
        with st.spinner("Thinking"):
            try:
                response=requests.post(
                    API_URL,
                    json={'question':question},
                    timeout=60
                )

                if response.status_code==200:
                    answer=response.json()['answer']
                    st.success("Answer")
                    st.write(answer)
                else:
                    st.error(f"API error:{response.status_code}")
                    st.text(response.text)
            except Exception as e :
                st.error('Could not connect to API')
                st.text(str(e))



