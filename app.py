# app.py
import os
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pypdf import PdfReader

# 1. โหลด environment variables จากไฟล์ .env
load_dotenv()

# 2. เริ่มต้น Qdrant (แบบ In-Memory)
qdrant_client = QdrantClient(":memory:")  # ใช้ In-Memory

# สร้าง Collection สำหรับเก็บเวกเตอร์
qdrant_client.recreate_collection(
    # กำหนดชื่อ Collection และ Config สำหรับเก็บเวกเตอร์ 
    collection_name="documents",
    # ใช้ vector ขนาด  384-D embedding
    # ใช่ระยะห่าง Cosine ในการคำนวณความคล้ายของเวกเตอร์
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  
    
)

# 3. ฟังก์ชันสำหรับอ่านไฟล์ PDF และแยกข้อความ
def extract_text_from_pdf(pdf_path):
    # อ่านข้อความจากไฟล์ PDF 
    reader = PdfReader(pdf_path)
    # รวมข้อความจากทุกหน้า
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 4. เตรียมข้อมูลเอกสารจากไฟล์ PDF
def prepare_documents_from_pdf(pdf_path):
    # อ่านข้อความจากไฟล์ PDF
    text = extract_text_from_pdf(pdf_path)
    # แยกข้อความด้วยบรรทัดใหม่
    documents = text.split("\n")  
    return [doc.strip() for doc in documents if doc.strip()]

# 5. แปลงข้อความเป็นเวกเตอร์ และเพิ่มลงใน Qdrant
def add_documents_to_qdrant(documents):
    # โหลดโมเดลสำหรับทำ Embedding
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  
    # แปลงข้อความเป็นเวกเตอร์
    vectors = embedding_model.encode(documents).tolist() 

    # เพิ่มข้อมูลลง Qdrant
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": documents[i]}) for i in range(len(documents))]
    # ใช้ Upsert เพื่อเพิ่มข้อมูลใหม่หรืออัปเดตข้อมูลเดิม 
    qdrant_client.upsert(collection_name="documents", points=points)

# 6. สร้างฟังก์ชันการค้นหาเอกสาร
def search_documents(query):
    # โหลดโมเดลสำหรับทำ Embedding  ใช้งาน Sentence Transforme ในการแปลงข้อความเป็นเวกเตอร์
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # แปลงคำถามเป็นเวกเตอร์
    query_vector = embedding_model.encode([query])[0].tolist()
    # ค้นหาเอกสารที่เกี่ยวข้อง
    search_results = qdrant_client.search(
        # เลือกใช้ Collection ที่เก็บเอกสาร
        collection_name="documents",
        # ใช้เวกเตอร์ของคำถามเป็น query_vector
        query_vector=query_vector,
        # ดึงเอกสารที่เกี่ยวข้อง 2 อันดับแรก สามารถเปลี่ยนค่าได้ n...
        limit=2  
    )
    return [hit.payload["text"] for hit in search_results]

# 7. สร้างฟังก์ชันการสร้างคำตอบด้วย Groq
def generate_answer(query):
    # ค้นหาข้อมูลที่เกี่ยวข้องจาก Qdrant
    retrieved_docs = search_documents(query)
    
    # รวมข้อมูลเข้าไปใน Prompt
    context = "\n".join(retrieved_docs) 
    # สร้าง Prompt สำหรับ Groq API โดยที่กำหนดข้อมูลอ้างอิงและคำถามที่ตอบมากๆ 
    prompt = [
        {"role": "system", "content": "คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับข้อมูลในเอกสาร จงตอบคำถามอย่างกระชับและถูกต้อง"},
        {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}\n\nคำตอบ:"}
    ]

    # เรียก Groq API สำหรับการสร้างคำตอบ 
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    # สร้างคำตอบจาก Groq API 
    response = groq_client.chat.completions.create(
        # set ค่า model  ใช้งาน llamas 3.3 70b สำหรับการสร้างคำตอบ 
        model="llama-3.3-70b-versatile",
        # กำหนดข้อมูล Prompt ที่สร้างไว้ 
        messages=prompt
    )
    # คืนคำตอบจาก Groq API
    return response.choices[0].message.content

# 8. สร้างอินเทอร์เฟซด้วย Streamlit
def main():
    # กำหนดชื่อและคำอธิบายของแอปพลิเคชัน
    st.title("RAG Chatbot เกี่ยวกับจังหวัดน่าน")
    # คำอธิบายเพิ่มเติม  
    st.write("สวัสดี Chatbot ที่ช่วยตอบคำถามจากเอกสารที่มีอยู่")

    # กำหนด path ของไฟล์ PDF
    pdf_path = "pdf/จังหวัดน่าน.pdf"

    # ตรวจสอบว่าไฟล์ PDF มีอยู่
    if os.path.exists(pdf_path):
        # อ่านข้อความจากไฟล์ PDF
        documents = prepare_documents_from_pdf(pdf_path)

        # เพิ่มข้อมูลลง Qdrant
        add_documents_to_qdrant(documents)
        st.success("เอกสาร PDF ถูกประมวลผลและพร้อมใช้งานแล้ว!")

        # รับคำถามจากผู้ใช้
        query = st.text_input("คุณ: ", placeholder="พิมพ์คำถามของคุณที่นี่...")
        
        # สร้างปุ่มสำหรับส่งคำถาม
        if st.button("ส่ง"):
            if query:
                # สร้างคำตอบ
                answer = generate_answer(query)
                # แสดงคำตอบ 
                st.write("Bot:", answer)
            else:
                # แสดงข้อความแจ้งเตือน 
                st.warning("กรุณาพิมพ์คำถามก่อนส่ง")
    else:
        st.error(f"ไม่พบไฟล์ PDF ที่ path: {pdf_path}")

# 9. เรียกใช้แอปพลิเคชัน
if __name__ == "__main__":
    main()