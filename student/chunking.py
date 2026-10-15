"""✏️ ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

เอกสารที่ได้รับเป็น markdown เช่น lecture note หรือ case study ที่มีหัวข้อ (#, ##), ตาราง และ code block
ชิ้นที่แบ่งแล้วจะถูกนำไปทำ embedding และใช้ค้นหาตอนตอบคำถาม
แบ่งดี = ค้นเจอเนื้อหาที่ตรงคำถามและครบถ้วน

ค่าตั้งต้นตอนยังไม่ทำ: ไม่แบ่งเลย ทั้งเอกสารเป็น chunk เดียว
ค่าที่ปรับได้ (CHUNK_SIZE, CHUNK_OVERLAP, ...) อยู่ใน student/settings.py
"""
import re
from student import settings


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int, header_prefix: str = "") -> list[str]:
    """ตัดข้อความที่ยาวเกิน chunk_size ออกเป็นหลายชิ้นตามย่อหน้า โดยมีการซ้อนกัน (overlap)"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_paras = []
    current_len = 0
    prefix_len = len(header_prefix)
    max_body_len = max(chunk_size - prefix_len, 100)

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue

        # กรณีย่อหน้าเดียวยาวเกิน max_body_len ให้ตัดย่อยตามบรรทัด
        if len(p) > max_body_len:
            if current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_len = 0

            lines = p.splitlines()
            sub_lines = []
            sub_len = 0
            for line in lines:
                l_str = line.strip()
                if not l_str:
                    continue
                if sub_len + len(l_str) + 1 > max_body_len and sub_lines:
                    chunks.append("\n".join(sub_lines))
                    overlap_txt = "\n".join(sub_lines)[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
                    sub_lines = [overlap_txt, l_str] if overlap_txt else [l_str]
                    sub_len = sum(len(x) + 1 for x in sub_lines)
                else:
                    sub_lines.append(l_str)
                    sub_len += len(l_str) + 1
            if sub_lines:
                chunks.append("\n".join(sub_lines))
            continue

        if current_len + len(p) + 2 > max_body_len and current_paras:
            combined = "\n\n".join(current_paras)
            chunks.append(combined)
            overlap_txt = combined[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            if overlap_txt:
                current_paras = [overlap_txt, p]
                current_len = len(overlap_txt) + len(p) + 2
            else:
                current_paras = [p]
                current_len = len(p)
        else:
            current_paras.append(p)
            current_len += len(p) + 2

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    result = []
    for c in chunks:
        c_str = c.strip()
        if not c_str:
            continue
        if header_prefix and not c_str.startswith(header_prefix.strip()):
            result.append(f"{header_prefix}{c_str}")
        else:
            result.append(c_str)

    return result or [text.strip()]


def _extract_heading_sections(text: str) -> list[tuple[str, str]]:
    """แยกเนื้อหาตามหัวข้อ Markdown (#, ##, ###, ...) คืนค่า list ของ (breadcrumb, content)"""
    lines = text.splitlines(keepends=True)
    sections = []
    current_headings = []  # stack ของ (level, title)
    current_lines = []
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        match = heading_re.match(line.strip())
        if match:
            if current_lines:
                content = "".join(current_lines).strip()
                if content:
                    breadcrumb = " > ".join(h[1] for h in current_headings)
                    sections.append((breadcrumb, content))
                current_lines = []

            level = len(match.group(1))
            title = match.group(2).strip()
            while current_headings and current_headings[-1][0] >= level:
                current_headings.pop()
            current_headings.append((level, title))
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        content = "".join(current_lines).strip()
        if content:
            breadcrumb = " > ".join(h[1] for h in current_headings)
            sections.append((breadcrumb, content))

    return sections


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ

    รับ:  doc_id  ชื่อเอกสาร เช่น "session04_01_prompt_fundamentals"
          text    เนื้อหาทั้งหมดของเอกสาร (markdown)
    คืน:  list[str] ข้อความของแต่ละ chunk (ห้ามว่าง)

    แนวทาง:
      - แยกตามหัวข้อ markdown (บรรทัดที่ขึ้นต้นด้วย #) แล้วใส่ชื่อหัวข้อไว้ต้น chunk ด้วย
      - หัวข้อที่ยาวเกินไปให้ตัดย่อยเป็นช่วง ๆ (เช่น 600–1,000 ตัวอักษร) และซ้อนกันเล็กน้อย (overlap)
      - พยายามอย่าตัดกลางตาราง กลาง code block หรือกลางประโยค
    ตัวอย่าง:
      split("doc1", "# A\nเนื้อหา A\n# B\nเนื้อหา B") -> ["A\nเนื้อหา A", "B\nเนื้อหา B"]
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return [doc_id]

    chunk_size = getattr(settings, "CHUNK_SIZE", 900)
    chunk_overlap = getattr(settings, "CHUNK_OVERLAP", 150)
    split_by_heading = getattr(settings, "SPLIT_BY_HEADING", True)
    include_heading = getattr(settings, "INCLUDE_HEADING", True)

    if not split_by_heading:
        return _split_long_text(cleaned_text, chunk_size, chunk_overlap)

    sections = _extract_heading_sections(cleaned_text)
    if not sections:
        return _split_long_text(cleaned_text, chunk_size, chunk_overlap)

    final_chunks = []
    for breadcrumb, content in sections:
        header_prefix = ""
        if include_heading and breadcrumb:
            if " > " in breadcrumb:
                header_prefix = f"[{breadcrumb}]\n\n"
            elif not content.startswith("#"):
                header_prefix = f"[{breadcrumb}]\n\n"

        if len(header_prefix) + len(content) <= chunk_size:
            chunk = f"{header_prefix}{content}".strip()
            if chunk:
                final_chunks.append(chunk)
        else:
            sub_chunks = _split_long_text(content, chunk_size, chunk_overlap, header_prefix)
            final_chunks.extend(sub_chunks)

    result = [c for c in final_chunks if c.strip()]
    return result if result else [cleaned_text]
