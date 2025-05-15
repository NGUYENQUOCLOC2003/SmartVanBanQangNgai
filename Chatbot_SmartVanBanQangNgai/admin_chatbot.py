import os
import json
import shutil
from dotenv import load_dotenv # type: ignore
import streamlit as st # type: ignore

from langchain_community.vectorstores import FAISS # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore
from langchain.chains import ConversationalRetrievalChain # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.document_loaders import ( # type: ignore
    PyPDFLoader, Docx2txtLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader
)

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VECTOR_DIR = os.path.join(working_dir, "vectorstore_data")
UPLOAD_DIR = os.path.join(working_dir, "upload")
LEARNED_FILE = os.path.join(working_dir, "learned_files.json")

st.set_page_config(page_title="📁 Quản trị SmartVănBảnQN", layout="wide")
st.title("📁 Quản lý tài liệu học cho Chatbot SmartVănBảnQN - Trợ lý Văn bản Quảng Ngãi Thông Minh")

# Cho phép upload trực tiếp
uploaded_files = st.file_uploader(
    label="📂 Tải tài liệu SmartVănBảnQN (PDF, Word, Excel...)",
    type=["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📥 Đang lưu {len(uploaded_files)} tài liệu vào thư mục upload/")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("✅ Đã lưu file vào thư mục upload. F5 để hệ thống học thêm.")

def load_learned_files():
    if os.path.exists(LEARNED_FILE):
        with open(LEARNED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_learned_files(file_list):
    with open(LEARNED_FILE, "w", encoding="utf-8") as f:
        json.dump(file_list, f, indent=2)

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".doc", ".docx"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    elif ext in [".ppt", ".pptx"]:
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"❌ Định dạng không hỗ trợ: {ext}")
    return loader.load()

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    doc_chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(doc_chunks, embeddings)

def create_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(llm=llm, output_key="answer", memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=False,
        return_source_documents=True
    )

learned_files = load_learned_files()
learned_index = {item["name"]: item["modified"] for item in learned_files}

new_files = []
if os.path.exists(UPLOAD_DIR):
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        modified_time = os.path.getmtime(file_path)
        if filename not in learned_index or learned_index[filename] != modified_time:
            new_files.append((filename, modified_time))
else:
    os.makedirs(UPLOAD_DIR)

if new_files:
    st.warning(f"🔍 Phát hiện {len(new_files)} tài liệu mới...")
    documents = []
    for filename, _ in new_files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            docs = load_document(file_path)
            documents.extend(docs)
            st.write(f"📄 Đã đọc: {filename}")
        except Exception as e:
            st.error(f"Lỗi khi đọc {filename}: {str(e)}")

    if documents:
        with st.spinner("📦 Đang xử lý và cập nhật FAISS..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
            new_vectorstore = setup_vectorstore(documents)

            if os.path.exists(VECTOR_DIR):
                existing = FAISS.load_local(
                    folder_path=VECTOR_DIR,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                existing.merge_from(new_vectorstore)
                existing.save_local(VECTOR_DIR)
            else:
                new_vectorstore.save_local(VECTOR_DIR)

            learned_files += [
                {"name": name, "modified": mod} for name, mod in new_files
            ]
            save_learned_files(learned_files)
        st.success("✅ Đã thêm dữ liệu mới vào bộ nhớ FAISS!")
else:
    st.success("✅ Không có tài liệu mới cần xử lý.")

if st.button("🗑️ Xóa toàn bộ dữ liệu đã học"):
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    if os.path.exists(LEARNED_FILE):
        os.remove(LEARNED_FILE)
    st.success("✅ Đã xóa toàn bộ dữ liệu học. Hãy tải lại tài liệu vào thư mục 'upload/' và F5 lại.")
