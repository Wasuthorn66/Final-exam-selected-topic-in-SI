"""✏️ ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

ค่าตั้งต้นตอนยังไม่ทำ: ไม่คืน chunk ใดเลย LLM จึงไม่เห็นเอกสาร
"""
import re
from core.types import Chunk
from core.utils import cosine, cosine_many
from student import settings


def _tokenize(text: str) -> set[str]:
    """แยกคำและ n-gram เพื่อใช้ในการทำ keyword search สำหรับภาษาไทยและภาษาอังกฤษ"""
    tokens = set()
    for word in re.findall(r"[\w\-]+", text.lower()):
        tokens.add(word)
        # สกัด n-gram 2-3 ตัวอักษรสำหรับภาษาไทย เพื่อช่วยจับคำที่ไม่มีเว้นวรรค
        thai_chars = [c for c in word if "\u0E00" <= c <= "\u0E7F"]
        if len(thai_chars) >= 2:
            for i in range(len(thai_chars) - 1):
                tokens.add("".join(thai_chars[i:i + 2]))
            for i in range(len(thai_chars) - 2):
                tokens.add("".join(thai_chars[i:i + 3]))
    return tokens


def make_query(question: str, options: list[str]) -> str:
    """สร้างข้อความที่จะใช้ค้นหา

    รับ:  question  คำถาม (หรือ requirement ของโจทย์เขียนโค้ด ซึ่งกรณีนี้ options จะเป็น list ว่าง)
          options   ข้อความของตัวเลือก ก–จ (ไม่มีตัวอักษร key นำหน้า)
    คืน:  สตริงคำค้น (ห้ามว่าง)

    แนวทาง: ลองรวมคำถามกับตัวเลือกเข้าด้วยกัน เพราะคำตอบมักอยู่ในย่อหน้าที่มีคำของตัวเลือก
    """
    q = question.strip()
    include_options = getattr(settings, "QUERY_WITH_OPTIONS", True)

    if include_options and options:
        opt_texts = [o.strip() for o in options if o and o.strip()]
        if opt_texts:
            return f"{q}\n" + "\n".join(opt_texts)

    return q if q else "query"


def search(chunks: list[Chunk], query_text: str, query_vector: list[float] | None, k: int) -> list[tuple[Chunk, float]]:
    """เลือก chunk ที่เกี่ยวข้องที่สุด k ชิ้น

    รับ:  chunks        chunk ทั้งหมดของเอกสารชุดนี้ (chunk.text, chunk.vector, chunk.doc_id)
          query_text    คำค้น (จาก make_query)
          query_vector  vector ของคำค้น (จาก embedding.embed_query) หรือ None ถ้ายังไม่ได้ทำ embedding
          k             จำนวนที่ระบบแนะนำ (5) จะใช้ settings.TOP_K ของตัวเองแทนก็ได้
    คืน:  list ของ (chunk, score) เรียงจากเกี่ยวข้องมากไปน้อย

    แนวทาง:
      - วิธีพื้นฐาน: scores = cosine_many(query_vector, [c.vector for c in chunks]) แล้วเลือก k ตัวที่คะแนนสูงสุด
      - วิธีที่ดีขึ้น: hybrid search คือรวมคะแนน vector กับคะแนนการจับคำ (keyword / BM25) ช่วยมากกับชื่อเฉพาะและตัวเลข
      - ระวังกรณีที่ chunk.vector หรือ query_vector เป็น None
    """
    if not chunks:
        return []

    top_k = getattr(settings, "TOP_K", k) or k or 5
    hybrid_weight = getattr(settings, "HYBRID_WEIGHT", 0.0)
    min_score = getattr(settings, "MIN_SCORE", None)

    # 1. คำนวณคะแนน Vector (Dense Similarity)
    has_vectors = query_vector is not None and any(c.vector is not None for c in chunks)
    vec_scores = [0.0] * len(chunks)

    if has_vectors:
        if all(c.vector is not None for c in chunks):
            vec_scores = cosine_many(query_vector, [c.vector for c in chunks])
        else:
            vec_scores = [
                cosine(query_vector, c.vector) if c.vector is not None else 0.0
                for c in chunks
            ]

    # 2. คำนวณคะแนน Keyword (Sparse Matching)
    kw_scores = [0.0] * len(chunks)
    if hybrid_weight > 0 or not has_vectors:
        q_tokens = _tokenize(query_text)
        if q_tokens:
            raw_kw_scores = []
            for c in chunks:
                c_tokens = _tokenize(c.text)
                intersection = q_tokens & c_tokens
                score = sum(1.5 if len(t) > 3 else 1.0 for t in intersection)
                raw_kw_scores.append(score)

            max_kw = max(raw_kw_scores) if raw_kw_scores else 0.0
            if max_kw > 0:
                kw_scores = [s / max_kw for s in raw_kw_scores]

    # 3. รวมคะแนน (Hybrid Score)
    scored_chunks: list[tuple[Chunk, float]] = []
    for i, c in enumerate(chunks):
        if has_vectors:
            if hybrid_weight > 0:
                final_score = (1.0 - hybrid_weight) * vec_scores[i] + hybrid_weight * kw_scores[i]
            else:
                final_score = vec_scores[i]
        else:
            final_score = kw_scores[i]

        if min_score is None or final_score >= min_score:
            scored_chunks.append((c, float(final_score)))

    scored_chunks.sort(key=lambda item: item[1], reverse=True)

    # ตัดซ้ำ (Deduplication): ป้องกัน chunk ที่มีเนื้อหาซ้อนทับกันมากเกินไป
    deduped_chunks: list[tuple[Chunk, float]] = []
    seen_prefixes = set()
    for c, score in scored_chunks:
        prefix_key = c.text.strip()[:120]
        if prefix_key in seen_prefixes:
            continue
        seen_prefixes.add(prefix_key)
        deduped_chunks.append((c, score))

    return deduped_chunks[:top_k]
