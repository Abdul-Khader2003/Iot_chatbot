from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_client import generate_sql_from_question, generate_human_response
from db_client import run_query
from predictor import predict_next_24_hours
from forecast import summarize_multiple_tags
import re
import pandas as pd
from datetime import datetime
import time
app = FastAPI()

class ChatRequest(BaseModel):
    question: str



@app.post("/chat")
def ask_bot(req: ChatRequest):
    try:
        start_time = time.time()
        sql_or_command = generate_sql_from_question(req.question)
        print("LLM output:", sql_or_command)
        tag_map = {"temp": "temp", "level": "level", "speed": "speed"}
        requested_tags = [tag_map[k] for k in tag_map if k in req.question.lower()]
        print(f"requested tags :{requested_tags}")

        if sql_or_command.startswith("PREDICT_NEXT_INTERVAL"):

            start_time3 = time.time()
            # Extract start and end timestamps using regex
            start_match = re.search(r"start=([\d\-T:]+)", sql_or_command)
            end_match = re.search(r"end=([\d\-T:]+)", sql_or_command)

            if not start_match or not end_match:
                raise ValueError("Invalid PREDICT_NEXT_INTERVAL format")

            start = pd.to_datetime(start_match.group(1))
            end = pd.to_datetime(end_match.group(1))
            end_time3 = time.time()

            print(f"prediction :{end_time3 - start_time3}")

            # Pass start and end to your prediction function
            start_time2 = time.time()
            print(f"start_time:{start_time2}")
            prediction = predict_next_24_hours(filter_tags=requested_tags, start=start, end=end)
            end_time2 = time.time()
            print(f"end_time:{end_time2}")
            print(f"predictor time taken :{end_time2 - start_time2}")
            print(prediction)

            start_time1 = time.time()
            result = summarize_multiple_tags(prediction, requested_tags)
            end_time1 = time.time()
            print(f"summary:{start_time1 - end_time1}")

            print(result)

            # Add start and end date/time to the result
            result_with_dates = {
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "data": result
            }

        else:
            data = run_query(sql_or_command)
            result_with_dates = {"past data": data}

        human_response = generate_human_response(req.question, result_with_dates)
        end_time = time.time()
        print(start_time - end_time)
        return {"answer": human_response,
                "time": start_time - end_time}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
