# services/azure_llm.py

import time
from openai import AzureOpenAI, RateLimitError
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

load_dotenv()
AZURE_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT=os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
class AzureLLMAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION
        )
        self.model = AZURE_DEPLOYMENT

    def complete(self, prompt: str, context: Any = None, max_retries: int = 3) -> str:
        
        for attempt in range(max_retries):
            try:
                messages = [{"role": "system", "content": "You are an AI data extractor."}]
                if context:
                    # attach context as system message (stringified if dict)
                    ctx = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
                    messages.append({"role": "system", "content": f"Context: {ctx}"})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2
                )
                return response.choices[0].message.content.strip()
            except RateLimitError:
                sleep_t = 2 ** attempt
                print(f"⚠️ Rate limit, sleeping {sleep_t}s then retrying...")
                time.sleep(sleep_t)
            except Exception as e:
                print(f"❌ LLM Error: {e}")
                if attempt == max_retries - 1:
                    return "NA"
                time.sleep(1)
        return "NA"



class RequestedField(BaseModel):
    field_name: str = Field(..., alias="fieldName")
    field_datatype: str = Field(..., alias="fieldDataType")
    field_desc: str = Field(..., alias="fieldDescription")
