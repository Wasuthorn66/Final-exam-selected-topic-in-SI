"""✏️ ส่วนที่ 2: Embedding — แปลงข้อความเป็น vector

ใช้ embed() จาก core เพื่อเรียกโมเดล embedding (ค่า embed_model ใน manifest.json)
    from core.ollama_client import embed
    embed(["ข้อความ 1", "ข้อความ 2"])  ->  [[0.12, -0.03, ...], [...]]

ค่าตั้งต้นตอนยังไม่ทำ: ไม่มี embedding เลย ค้นหาด้วยความหมายไม่ได้
"""
from core.ollama_client import embed
from student import settings


def embed_documents(texts: list[str]) -> list[list[float]]:
    """แปลง chunk ทั้งหมดของเอกสารเป็น vector (ถูกเรียกครั้งเดียวตอนรับ knowledge)

    รับ:  texts  list ของข้อความ chunk (ผลจาก chunking.split)
    คืน:  list ของ vector จำนวนเท่ากับ texts และเรียงลำดับเดียวกัน

    แนวทาง:
      - เรียก embed(texts) ได้เลย
      - โมเดล nomic-embed-text ทำงานดีขึ้นถ้าเติม prefix "search_document: " หน้าข้อความเอกสาร
    """
    if not texts:
        return []

    prefix = getattr(settings, "DOC_PREFIX", "search_document: ")
    prefixed_texts = [f"{prefix}{t}" for t in texts]
    return embed(prefixed_texts)


def embed_query(text: str) -> list[float]:
    """แปลงคำค้น 1 ข้อความเป็น vector (ถูกเรียกทุกครั้งที่ถามคำถาม)

    รับ:  text  คำค้น (ผลจาก retrieval.make_query)
    คืน:  vector 1 ตัว (list[float])

    แนวทาง:
      - embed([text])[0]
      - ฝั่งคำค้นของ nomic-embed-text ใช้ prefix "search_query: "
    """
    prefix = getattr(settings, "QUERY_PREFIX", "search_query: ")
    prefixed_text = f"{prefix}{text}"
    vectors = embed([prefixed_text])
    return vectors[0]
