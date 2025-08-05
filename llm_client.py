from datetime import  timedelta

from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime

today_str = datetime.now().strftime("%Y-%m-%d")
now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
load_dotenv()

# Initialize SambaNova-compatible client
client = OpenAI(
    api_key=os.getenv("SAMBA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

SYSTEM_PROMPT = f"""
You are an AI assistant that converts natural language questions into either:

1. SQL queries for a ClickHouse table TEST.tag_data_30mins with columns:
   - tagName (String): values like 'temp', 'speed', 'level'
   - tagAddress (Int)
   - createdAt (DateTime)
   - value (Float)

2. OR, if the question asks about future values or prediction, you must return one of these exact commands without any markdown, explanation, or formatting:

   - For exactly next 24 hours prediction:
     PREDICT_NEXT_24_HOURS

   - For custom future intervals, respond with:
     PREDICT_NEXT_INTERVAL start=YYYY-MM-DDTHH:MM:SS end=YYYY-MM-DDTHH:MM:SS

   The timestamps must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS), in 24-hour time.

   IMPORTANT: If the question mentions a specific **future date or date range**, use that exact date or date range in the command.
   - If the user asks for a **specific time only** (e.g., "at 05:00", "at 3pm"), assume a 1-hour window starting from that time (e.g., 05:00 to 06:00) **unless the user specifies otherwise**.

   🕒 Today's date is: {today_str}
   🕒 Current time is: {now_iso}
   
   Examples:

   Q: "Predict average level on 2025-08-06"
   A: PREDICT_NEXT_INTERVAL start=2025-08-06T00:00:00 end=2025-08-06T23:59:59

   Q: "Predict temperature from 2025-08-06 10:00 to 2025-08-07 10:00"
   A: PREDICT_NEXT_INTERVAL start=2025-08-06T10:00:00 end=2025-08-07T10:00:00
   
   Q: "Predict temperature on 5:00pm" 
   (if the values are not present it needs to give the next nearest predicted value to it)
   A: PREDICT_NEXT_INTERVAL start=2025-08-04 17:17:59 end=2025-08-04 17:17:59
   
   

Return ONLY the SQL query or the exact command string.
"""

def clean_sql(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```sql"):
        sql = sql.replace("```sql", "").replace("```", "").strip()
    elif sql.startswith("```"):
        sql = sql.replace("```", "").strip()
    return sql.strip().rstrip(";")

def generate_sql_from_question(question: str) -> str:
    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        top_p=0.95,
    )
    raw_sql = response.choices[0].message.content
    return clean_sql(raw_sql)

# 🔹 Prompt for predictions (includes reasoning)
PREDICT_PROMPT = f"""
You are an assistant that helps summarize raw database values into human-readable form.

Given a user's question and exact past values (e.g., last temperature or speed reading), explain what the data shows in plain English.

Do not perform any calculations or summarization — just express the values as-is.

🕒 Today's date is: {today_str}
🕒 Current time is: {now_iso}

⚠️ Important:
When referencing any predictions or time windows, **use the current time (`now_iso`) to calculate or express the relevant timeframe.**
Avoid using hardcoded times like 00:00 unless they align with the current `now_iso`.
"""

# 🔹 Prompt for past data (no calculations, just rephrase)
PAST_PROMPT = """
You are an assistant that helps summarize raw database values into human-readable form.
Given a user's question and exact past values (e.g., last temperature or speed reading), explain what the data shows in plain English.
Do not perform any calculations or summarization — just express the values as-is.
"""

def generate_human_response(question: str, result: dict) -> str:
    print(result)
    today_str = datetime.now().strftime("%Y-%m-%d")
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    prompt = f"Today is {today_str}, and tomorrow is {tomorrow_str}. " \
             f"User's question: {question}\n" \
             f"System response: {result}"

    # 🧠 Choose correct system prompt based on data type
    if "predict" in result:
        system_prompt = PREDICT_PROMPT
    elif "past data" in result:
        system_prompt = PAST_PROMPT
    else:
        system_prompt = "You are a helpful assistant. Rephrase the result in simple words."

    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        top_p=0.95
    )

    return response.choices[0].message.content.strip()