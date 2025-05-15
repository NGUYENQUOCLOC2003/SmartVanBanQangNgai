import os
from dotenv import load_dotenv # type: ignore
import streamlit as st # type: ignore

from langchain_community.vectorstores import FAISS # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore
from langchain.chains import ConversationalRetrievalChain # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VECTOR_DIR = os.path.join(working_dir, "vectorstore_data")

st.set_page_config(page_title="🤖 Chatbot SmartVănBảnQN - Trợ lý Văn bản Quảng Ngãi Thông Minh", layout="wide")
st.title("🤖 ChatbotSmartVănBảnQN - Trợ lý Văn bản Quảng Ngãi Thông Minh")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if os.path.exists(VECTOR_DIR):
    with st.spinner("🚀 Đang tải dữ liệu đã học vui lòng chờ vài giây..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            folder_path=VECTOR_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        memory = ConversationBufferMemory(llm=llm, output_key="answer", memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            memory=memory,
            return_source_documents=True
        )

        st.session_state.conversation_chain = chain
else:
    st.error("❌ Chưa có dữ liệu học. Vui lòng liên hệ quản trị viên để cập nhật tài liệu.")

# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
user_input = st.chat_input("🧮 Hỏi gì về văn bản Quảng Ngãi nào?")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("💬 Đang truy xuất và suy luận..."):
            try:
                if "conversation_chain" not in st.session_state:
                    raise Exception("❌ Dữ liệu chưa sẵn sàng.")

                response = st.session_state.conversation_chain.invoke({"question": user_input})
                source_docs = response.get("source_documents", [])

                if not source_docs:
                    assistant_response = "❌ Câu hỏi này không có trong tài liệu đã học. Vui lòng hỏi lại theo nội dung Toán 12."
                else:
                    assistant_response = response.get("answer", "❌ Không có câu trả lời phù hợp.")
            except Exception as e:
                assistant_response = str(e)

            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
