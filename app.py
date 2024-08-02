import os
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Load environment variables from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Define data directory
data_directory = "data"

# Function to create LLM response
def llm(user_input):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": 
             """
             Sen Türkiye'de bir avukatsın ismin Ayse Yilmaz. Aşağıdaki dava için bir savunma dilekçesi yaz gereken bosluklari doldur. Dilekçede savunmanın özeti, sanık ve müdafi bilgileri, hukuki dayanaklar ve talep edilen kararlar yer almalıdır.
             Örnek savunma dilekçesi:
             
             SAVUNMA DİLEKÇESİ ÖRNEĞİ 
             …… MAHKEMESİNE
             
             DOSYA NO                    : …… 
             
             SANIK                            : …… 
             
             MÜDAFİ                        : …… 
             
             KONU                            : Savunma dilekçesi 
             
             AÇIKLAMALARIMIZ          
             
             Müvekkil …… suçsuz olup; beraatine karar verilmesi gerekmektedir. Şöyle ki; 
             
             1- Müvekkil, uyuşturucu madde ele geçirilen aracın sahibi olmayıp; araç …… ismine kayıtlıdır ve aracı fiilen …… isimli kişi kullanmaktadır.
              
             2- Uyuşturucu maddelerin söz konusu araca müvekkil tarafından konulduğuna ilişkin hiçbir somut delil bulunmamaktadır. Suç konusu uyuşturucu maddelerde ve aracın torpido kısmında müvekkilin parmak izi ve benzeri hiçbir delil bulunmamaktadır. 
             
             3- Masumiyet karinesi uyarınca da müvekkile beraat kararı verilmesi gerekmektedir. 
             
             4- Müvekkilin kanında herhangi bir uyuşturucu maddeye de rastlanılamamış olup; benzeri bir suçtan sabıkası bulunmamaktadır. Ayrıca yapılan arama hukuka aykırıdır. 
             
             Yukarıda açıklanan tüm sebeplerle müvekkile beraat kararı verilmesini talep etme zorunluluğu hâsıl olmuştur. 
             
             SONUÇ VE İSTEM       : Müvekkilin iddia edilen suçu işlediğine ilişkin olarak hiçbir somut delil bulunmaması ve müvekkilin suçsuz olması sebebiyle BERAATİNE karar verilmesini saygılarımızla arz ve talep ederiz. (Tarih: …/…/………) 
             """
            },
            {"role": "user", "content": f" Bu davaya gore sanik bilgireini dilekceye girin Dava: {user_input}. "}
        ],
        temperature=0.3,
        max_tokens=5000,
        seed=45
    )
    return response.choices[0].message.content.strip()


embeddings = OpenAIEmbeddings()

# Save uploaded files
def save_uploaded_files(uploaded_files):
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return len(uploaded_files)

# Data ingestion function
def data_ingestion():
    loader = PyPDFLoader(data_directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# Streamlit front end
st.title("AI Türk Paralegal")

# Text input for case details
dava_detayi = st.text_area("Dava Detaylarını Girin:", """

""")

# File uploader for PDFs
uploaded_files = st.file_uploader("PDF Dosyaları Yükleyin:", accept_multiple_files=True)

# Button to save files
if st.button("Dosyaları Kaydet"):
    if uploaded_files:
        num_files_saved = save_uploaded_files(uploaded_files)
        st.success(f"{num_files_saved} dosya başarıyla kaydedildi.")
    else:
        st.warning("Lütfen en az bir dosya yükleyin.")

# Button to ingest data
if st.button("Verileri İşle"):
    docs = data_ingestion()
    get_vector_store(docs)
    st.success("Veriler başarıyla işlendi ve vektör veri tabanına kaydedildi.")

# Button to generate legal document
if st.button("Dilekçe Oluştur"):
    if dava_detayi:
        dilekce = llm(dava_detayi)
        st.subheader("Oluşturulan Dilekçe:")
        st.write(dilekce)
    else:
        st.warning("Lütfen dava detaylarını girin.")
