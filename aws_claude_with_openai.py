import base64
import os
from dotenv import load_dotenv
import json
from azure_llm import AzureLLMAgent
from typing import List, Dict
from marksheet_field import fields
import re
from collections import defaultdict
load_dotenv()
import boto3
from dotenv import load_dotenv
load_dotenv()

#print(client.models.list())
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "ap-south-1"
AWS_CLAUDE_MODEL_ID=os.getenv("AWS_CLAUDE_MODEL_ID")

agent = AzureLLMAgent()

def extract_all_fields_and_tables(field_schema: List[Dict],
                                  content: str,
                                  agent,
                                  context=None):

    # ---- Split schemas ----
    field_items = [f for f in field_schema if f["fieldType"] == "field"]
    table_items = [f for f in field_schema if f["fieldType"] == "table"]

    # ---- Build field instructions ----
    field_text_lines = []
    for f in field_items:
        extra = (
            f"with {f['fieldName']} - carefully understand what is meant by "
            f"the description ({f['fieldDescription']}) for this field and return "
            f"the output in the type expected ({f['fieldDatatype']}) "
            "(e.g., string, number, date, etc.)."
        )
        field_text_lines.append(
            f"- **{f['fieldName']}** ({f['fieldDatatype']}): {f['fieldDescription']}\n  {extra}"
        )
    field_text = "\n".join(field_text_lines)

    # ---- Build table instructions grouped by tableName ----
    tables = defaultdict(list)
    for r in table_items:
        tables[r["tableName"]].append(r)

    table_text = ""
    for table_name, cols in tables.items():
        table_text += f"\n### Table: {table_name}\n"
        for c in cols:
            extra = (
                f"with {c['fieldName']} - carefully understand what is meant by "
                f"the description ({c['fieldDescription']}) for this column and return "
                f"the output in the type expected ({c['fieldDatatype']}) "
                "(e.g., string, number, date, etc.)."
            )
            table_text += f"- **{c['fieldName']}** ({c['fieldDatatype']}): {c['fieldDescription']}\n  {extra}\n"

    # ---- Build universal prompt ----
    prompt = f"""
You are an extremely accurate information extraction model.

Extract ALL requested fields and ALL table rows across ALL pages.

========================================================
FIELDS TO EXTRACT (OUTPUT AS SIMPLE KEY: VALUE)
========================================================
{field_text}

========================================================
TABLES TO EXTRACT (OUTPUT AS ITEMS ARRAY)
========================================================
(Return format example:
"TableName": {{
    "fieldType": "table",
    "items": [
        {{ "col1": "...", "col2": "..." }},
        ...
    ]
}}
)

{table_text}

========================================================
STRICT RULES
========================================================
- Return value in EXACT datatype expected.
- If no value found → return null.
- Preserve field and table names EXACTLY.
- Do NOT rename or modify keys.
- Output ONLY valid JSON.
- No explanation. No markdown.

========================================================
CONTENT
========================================================
{content}

Return FINAL JSON ONLY:
""".strip()

    # ---- Call LLM ----
    try:
        raw = agent.complete(prompt, context=context) if context else agent.complete(prompt)
    except Exception as e:
        print("❌ LLM failed:", e)
        return None

    # ---- Extract JSON safely ----
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw)
    except Exception as e:
        print("❌ JSON parsing error:", e)
        return {"raw": raw}
    


def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )   



def claude_ocr(file_path: str) -> str:
    # Read file and encode to base64
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    encoded_file = base64.b64encode(file_bytes).decode("utf-8")

    # Detect media type
    if file_path.lower().endswith(".pdf"):
        media_type = "application/pdf"
    elif file_path.lower().endswith(".png"):
        media_type = "image/png"
    elif file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg"):
        media_type = "image/jpeg"
    else:
        raise ValueError("Unsupported file type")

    # Build the message content
    message_content = [
        {
            "type": "document" if media_type == "application/pdf" else "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded_file,
            },
        },
        {
            "type": "text",
            "text": """
                        Extract all text from this document exactly as written. Preserve tables, line breaks, and formatting as much as possible.
                        CRITICAL RULES FOR NUMERIC FIELDS (marks, scores, roll numbers, register numbers, ID numbers, totals, grades expressed as numbers):

                        In handwritten or low-quality scanned documents, certain characters are commonly misread. Apply the following corrections ONLY in positions where a digit is expected (numeric fields):

                                - "O" or "o" → 0  (letter O confused with zero)
                                - "D" → 0  (letter D confused with zero)
                                - "Q" → 0  (letter Q confused with zero)
                                - "b" → 6  (letter b confused with six)
                                - "S" or "s" → 5  (letter S confused with five)
                                - "I" or "l" or "|" or "\" → 1  (letter I/l or pipe/backslash confused with one)
                                - "Z" or "z" → 2  (letter Z confused with two)
                                - "B" → 8  (letter B confused with eight)
                                - "G" → 6  (letter G confused with six)
                                - " " (space within a number) → remove the space (e.g. "1 1 0" → "110")

                                HOW TO IDENTIFY NUMERIC FIELDS:
                                - Any column or field labeled: Marks, Score, Total, Max, Min, Grade (if numeric), Roll No, Register No, Reg No, ID, Enrollment No, Seat No, Percentage, Rank, Year, Age.
                                - Any cell in a marks table where surrounding cells contain digits.
                                - Multi-digit sequences that are clearly IDs or roll numbers.

                                DO NOT apply these corrections to:
                                - Names of students, subjects, schools, or places.
                                - Text fields, addresses, or descriptions.
                                - Single letters that represent grades (e.g. A, B, C, F) unless they are clearly a misread digit in a numeric column.

                                Return the corrected, clean extracted text.
                        
            
            """
            
        },
    ]

    # Call Claude via Amazon Bedrock
    bedrock_client = get_bedrock_client()

    response = bedrock_client.invoke_model(
        modelId=AWS_CLAUDE_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        })
    )

    result = json.loads(response['body'].read())
    return result['content'][0]['text']



result = claude_ocr("Marksheet(mountzion)\kalai_1.jpeg")  

#file_path='mugundhan_1.jpeg'
#result= extract_text_from_file(file_path)

print('result',result)


final_result = extract_all_fields_and_tables(fields, result, agent)

print('final result',final_result)