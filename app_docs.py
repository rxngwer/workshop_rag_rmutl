# app.py
import os
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. โหลด environment variables จากไฟล์ .env
load_dotenv()

# 2. เริ่มต้น Qdrant (แบบ In-Memory)
qdrant_client = QdrantClient(":memory:")  # ใช้ In-Memory

# สร้าง Collection สำหรับเก็บเวกเตอร์
qdrant_client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # ใช้ 384-D embedding
)

# 3. ข้อมูลเอกสารที่กำหนดเอง
documents = [
    "น่าน (ไทยถิ่นเหนือ: ᨶᩣ᩠᩵ᨶ) เป็นจังหวัดหนึ่งในประเทศไทย",
    "ตั้งอยู่ทางทิศตะวันออกสุดของภาคเหนือ และมีความสำคัญในประวัติศาสตร์ไทย",
    "เป็นที่ตั้งของเมืองที่สำคัญในอดีต เช่น เวียงวรนคร (เมืองพลัว) เวียงศีรษะเกษ (เมืองงั่ว), และเวียงภูเพียงแช่แห้ง",
    "จังหวัดน่านยังเป็นแหล่งต้นน้ำของแม่น้ำน่าน ซึ่งไหลผ่าน",
    "ทำให้มีทรัพยากรธรรมชาติที่อุดมสมบูรณ์และทิวทัศน์ที่สวยงาม",
    "จังหวัดน่านยังมีวัฒนธรรมและประเพณีที่หลากหลาย อันเป็นเอกลักษณ์ของพื้นที่",
    "ทำให้เป็นจุดหมายที่น่าสนใจสำหรับนักท่องเที่ยวทั้งชาวไทยและต่างประเทศ"
]

# 4. แปลงข้อความเป็นเวกเตอร์ และเพิ่มลงใน Qdrant
def add_documents_to_qdrant(documents):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # โหลดโมเดลสำหรับทำ Embedding
    vectors = embedding_model.encode(documents).tolist()  # แปลงข้อความเป็นเวกเตอร์

    # เพิ่มข้อมูลลง Qdrant
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": documents[i]}) for i in range(len(documents))]
    qdrant_client.upsert(collection_name="documents", points=points)

# 5. สร้างฟังก์ชันการค้นหาเอกสาร
def search_documents(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedding_model.encode([query])[0].tolist()
    search_results = qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=2  # ดึงเอกสารที่เกี่ยวข้อง 2 อันดับแรก
    )
    return [hit.payload["text"] for hit in search_results]

# 6. สร้างฟังก์ชันการสร้างคำตอบด้วย Groq
def generate_answer(query):
    # ค้นหาข้อมูลที่เกี่ยวข้องจาก Qdrant
    retrieved_docs = search_documents(query)

    # รวมข้อมูลเข้าไปใน Prompt
    context = "\n".join(retrieved_docs)
    prompt = [
        {"role": "system", "content": "คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับข้อมูลในเอกสาร จงตอบคำถามอย่างกระชับและถูกต้อง"},
        {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}\n\nคำตอบ:"}
    ]

    # เรียก Groq API
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=prompt
    )

    return response.choices[0].message.content

# 7. สร้างอินเทอร์เฟซด้วย Streamlit
def main():
    st.title("RAG Chatbot สำหรับข้อมูลจังหวัดน่าน")
    st.write("สวัสดี! ฉันคือ Chatbot ที่ช่วยตอบคำถามเกี่ยวกับจังหวัดน่าน")

    # เพิ่มข้อมูลเอกสารลงใน Qdrant
    add_documents_to_qdrant(documents)
    st.success("ข้อมูลเอกสารพร้อมใช้งานแล้ว!")

    # รับคำถามจากผู้ใช้
    query = st.text_input("คุณ: ", placeholder="พิมพ์คำถามของคุณที่นี่...")

    if st.button("ส่ง"):
        if query:
            # สร้างคำตอบ
            answer = generate_answer(query)
            st.write("Bot:", answer)
        else:
            st.warning("กรุณาพิมพ์คำถามก่อนส่ง")

# 8. เรียกใช้แอปพลิเคชัน
if __name__ == "__main__":
    main()