"""✏️ ส่วนที่ 4: Prompt — บอก LLM ว่าจะให้ทำอะไร

คืนค่าเป็น list ของ message แบบ chat:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
(หรือคืนเป็นสตริงก็ได้ ระบบจะถือว่าเป็นข้อความของ user)

hits คือผลจาก retrieval.search เป็น list ของ (chunk, score) อ่านเนื้อหาได้จาก chunk.text และ chunk.doc_id

ค่าตั้งต้นตอนยังไม่ทำ: ไม่เรียก LLM เลย ข้อสอบปรนัยตอบ "ก" ทุกข้อ และโจทย์โค้ดได้ฟังก์ชันว่างที่ raise NotImplementedError
(ทำส่วนนี้ก่อน เพราะถ้ายังไม่ทำ ส่วนอื่นจะไม่มีผลต่อคะแนนเลย)
"""
from student import settings


def _format_context(hits, max_chars: int) -> list[str]:
    """จัดรูปแบบ chunks ที่ค้นได้เป็นข้อความอ้างอิง โดยจำกัดความยาวไม่เกิน max_chars"""
    if not hits:
        return []
    context_blocks = []
    total_chars = 0
    for item in hits:
        chunk = item[0] if isinstance(item, (tuple, list)) else item
        text = getattr(chunk, "text", str(chunk)).strip()
        doc_id = getattr(chunk, "doc_id", "")
        header = f"[เอกสาร: {doc_id}]" if doc_id else "[เอกสารอ้างอิง]"
        block = f"{header}\n{text}"

        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                context_blocks.append(block[:remaining] + "\n...(ข้อความบางส่วนถูกตัด)")
            break
        context_blocks.append(block)
        total_chars += len(block)
    return context_blocks


def build_mc_prompt(question: str, options: list[tuple[str, str]], hits) -> list[dict]:
    """prompt สำหรับข้อสอบปรนัย

    รับ:  question  คำถาม
          options   list ของ (key, text) เช่น [("ก", "O(1)"), ("ข", "O(log n)"), ...]
          hits      chunk ที่ค้นได้ (list ของ (chunk, score)) อาจเป็น list ว่าง
    คืน:  list ของ message

    หมายเหตุ: core บังคับให้ LLM ตอบเป็น JSON {"reason": "...", "answer": "<key>"} ให้อยู่แล้ว
    แนวทาง:
      - ใส่เนื้อหาของ hits เป็น "เอกสารอ้างอิง" แล้วสั่งให้ยึดเอกสารเป็นหลัก แม้จะขัดกับความรู้ทั่วไป
      - แสดงตัวเลือกพร้อม key (ก. ... ข. ...)
      - ระวังคำถามเชิงปฏิเสธ เช่น "ข้อใด**ไม่**ถูกต้อง"
    """
    max_chars = getattr(settings, "MAX_CONTEXT_CHARS", 6000)
    context_blocks = _format_context(hits, max_chars)

    system_content = (
        "คุณเป็นผู้เชี่ยวชาญในการตอบข้อสอบปรนัย (Multiple Choice) เกี่ยวกับการเขียนโปรแกรมและระบบคอมพิวเตอร์\n"
        "หน้าที่ของคุณคือวิเคราะห์คำถามและตัวเลือกอย่างละเอียดถี่ถ้วน แล้วตอบคำถามให้ถูกต้องที่สุด\n\n"
        "กฎสำคัญที่ต้องปฏิบัติตามอย่างเคร่งครัด:\n"
        "1. หากมี 'เอกสารอ้างอิง' ให้ยึดข้อมูลและข้อเท็จจริงในเอกสารอ้างอิงเป็นหลักอย่างเคร่งครัดที่สุด แม้จะขัดแย้งกับความรู้ทั่วไปภายนอก\n"
        "2. อ่านหัวตารางและกติกาอย่างรอบคอบ: เช่น 'ยอดซื้อขั้นต่ำ' หมายถึงเกณฑ์ซื้อขั้นต่ำ และ 'ส่วนลดสูงสุดต่อบิล' หมายถึงเพดานจำนวนเงินส่วนลด ไม่ใช่เพดานยอดซื้อ หากคำนวณส่วนลดแล้วเกินเพดาน ให้ได้ส่วนลดเท่ากับเพดานส่วนลดสูงสุด\n"
        "3. สังเกตคำถามเชิงปฏิเสธหรือเงื่อนไขพิเศษอย่างระมัดระวัง เช่น 'ข้อใดไม่ถูกต้อง', 'ข้อใดไม่ใช่', 'ยกเว้น', 'ผิด'\n"
        "4. วิเคราะห์ตัวเลือกทุกข้ออย่างรอบคอบและเลือกตัวเลือกที่ตรงกับคำถามที่สุดเพียงข้อเดียว\n"
        "5. ต้องตอบในรูปแบบ JSON ที่มี 2 ฟิลด์:\n"
        '   - "reason": วิเคราะห์และอธิบายเหตุผลสั้น ๆ ว่าทำไมตัวเลือกนี้จึงถูกต้อง\n'
        '   - "answer": ระบุเฉพาะ key ของตัวเลือกที่ถูกต้องเพียง 1 ตัวเลือกเท่านั้น (เช่น "ก", "ข", "ค", "ง")'
    )

    formatted_options = []
    for key, text in options:
        t = text.strip()
        if not (t.startswith(f"{key}.") or t.startswith(f"{key})") or t.startswith(f"{key} ")):
            formatted_options.append(f"{key}. {t}")
        else:
            formatted_options.append(t)
    options_str = "\n".join(formatted_options)
    allowed_keys_str = ", ".join(f'"{k}"' for k, _ in options)

    user_parts = []
    if context_blocks:
        user_parts.append("### เอกสารอ้างอิง:\n" + "\n\n".join(context_blocks))

    user_parts.append(f"### คำถาม:\n{question.strip()}")
    user_parts.append(f"### ตัวเลือก:\n{options_str}")
    user_parts.append(
        f"จงวิเคราะห์และเลือกคำตอบที่ถูกต้องที่สุดจาก {allowed_keys_str}\n"
        'ตอบเป็น JSON เท่านั้น: {"reason": "...", "answer": "..."}'
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_code_prompt(requirement: str, signature: str | None, entry_point: str | None, hits) -> list[dict]:
    """prompt สำหรับโจทย์เขียนโปรแกรม

    รับ:  requirement  โจทย์ (ภาษาไทย)
          signature    บรรทัด def ที่ต้องใช้ เช่น "def is_prime(n: int) -> bool:"
          entry_point  ชื่อฟังก์ชัน เช่น "is_prime"
          hits         chunk ที่ค้นได้ (มีเฉพาะโจทย์ที่อ้างถึงเอกสาร ส่วนใหญ่จะเป็น list ว่าง)
    คืน:  list ของ message

    หมายเหตุ: core ดึงโค้ดจาก ```python ... ``` ในคำตอบให้ และถ้า compile ไม่ผ่านจะขอให้ LLM แก้ให้อีก 1 รอบ
    แนวทาง:
      - สั่งให้ใช้ชื่อฟังก์ชันและ signature ตามที่ให้มาเป๊ะ ๆ และใช้เฉพาะ Python standard library
      - ห้าม print หรือรับ input เอง ให้ return ค่า
      - อ่านโจทย์ให้ดี บางข้อ "ห้ามใช้" ฟังก์ชันบางตัว (เช่น max, sorted, eval) ถ้าใช้จะได้ 0 ทั้งข้อ
      - ให้ตอบเป็น code block เดียว
    """
    max_chars = getattr(settings, "MAX_CONTEXT_CHARS", 6000)
    context_blocks = _format_context(hits, max_chars)

    system_content = (
        "You are an expert Python software engineer.\n"
        "Your task is to write clean, correct, and efficient Python code based on the given requirements.\n\n"
        "CRITICAL RULES:\n"
        "1. Response format: Respond ONLY with a single ```python ... ``` code block containing the complete Python solution. Do not write any explanations or conversational text outside the code block.\n"
        "2. Function interface: You MUST use the exact function signature and entry point name provided. The function must RETURN the expected value. NEVER use print() or input().\n"
        "3. No top-level execution: Do NOT include example calls, testing code, or print() at the module level. Only define the required function and any necessary helpers or imports.\n"
        "4. Imports & Standard library: If your code uses any standard library module (such as 'import re' or 'import math'), you MUST explicitly write the import statement at the top of the code block. Never use 're' or other modules without importing them. Do not import third-party packages.\n"
        "5. Respect constraints: Carefully check if any functions or methods are forbidden by the problem (e.g., forbidden max, min, sorted, eval). Do NOT use any forbidden functions.\n"
        "6. Business rules & string patterns:\n"
        "   - Follow all specifications in reference documents strictly.\n"
        "   - When validating patterns, use re.fullmatch or exact regex according to the document specs (e.g. if code must start with uppercase/digits or prefix, convert code to uppercase first if specified, e.g. `code.upper()`).\n"
        "   - Verify pattern lengths and indices against document examples (e.g. `BB-` (3 chars) + 4 alphanum + `-` (1 char) + 2 digits = total 10 chars, regex `r'^BB-[A-Z0-9]{4}-\\d{2}$'`). Do not miscalculate string lengths.\n"
        "   - In discount/table calculations: Each coupon/rule has its OWN specific minimum spend ('ยอดซื้อขั้นต่ำ') and maximum discount cap ('ส่วนลดสูงสุดต่อบิล') from the table. Do NOT hardcode a single minimum spend across all items! Structure your lookup with named keys (e.g. `rules = {'CODE': {'min_spend': ..., 'type': ..., 'val': ..., 'max_cap': ...}}`) or clear if/elif branches to avoid tuple indexing errors. For each coupon, check `if total < item_min_spend: return 0.0`. If cash discount, discount cannot exceed total. Return float rounded to 2 decimal places."
    )

    user_parts = []
    if context_blocks:
        user_parts.append("### เอกสารอ้างอิง / ข้อกำหนดธุรกิจ:\n" + "\n\n".join(context_blocks))

    user_parts.append(f"### โจทย์:\n{requirement.strip()}")

    if signature:
        user_parts.append(f"### Function Signature ที่ต้องใช้ (ต้องตรงทุกตัวอักษร):\n```python\n{signature}\n```")
    elif entry_point:
        user_parts.append(f"### ชื่อฟังก์ชันที่ต้องใช้:\n`{entry_point}`")

    user_parts.append(
        "### ข้อกำหนดในการส่งมอบ:\n"
        "- ให้ตอบเป็น code block เดียวเท่านั้น: ```python ... ```\n"
        "- หากใช้โมดูลมาตรฐาน (เช่น re) ต้องใส่คำสั่ง import (เช่น import re) ไว้บรรทัดแรกใน code block เสมอ\n"
        "- ฟังก์ชันต้อง return ผลลัพธ์ ห้าม print() หรือ input()\n"
        "- จัดการ edge cases ให้ครบถ้วน (เช่น ค่า 0, None, ข้อมูลว่าง, ตัวพิมพ์เล็ก/ใหญ่)\n"
        "- ห้ามใส่โค้ดทดสอบเรียกใช้ฟังก์ชันนอก def"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
