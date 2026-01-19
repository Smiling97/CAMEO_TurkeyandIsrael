import json
import os
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import pandas as pd
from openai import OpenAI

# ==================== Configuration ==================== #
# Recommended models: "gpt-4o", "gpt-4-turbo", or "gpt-3.5-turbo-0125"
OPENAI_MODEL_NAME = "gpt-4.1"

DATA_CSV_PATH = Path("ThemarkerHebrew.csv")
CONTENT_COLUMN = "Content"
NEWS_ID_COLUMN = "NewsID"

# Optional Input Columns
SOURCE_COLUMN = "Source"
DATE_COLUMN = "Date"
TITLE_COLUMN = "Title"

OUTPUT_CSV_PATH = Path("Societal_Sentiment_Israel_OpenAI.csv")

OUTPUT_LANGUAGE = "English"

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable missing.")

# Initialize Standard OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

print(f"Using OpenAI Model: {OPENAI_MODEL_NAME}")


# ==================== Helpers ==================== #
def _language_clause():
    return "Respond in Turkish." if OUTPUT_LANGUAGE.lower().startswith("turk") else "Respond in English."


def call_with_retries(fn: Callable, max_attempts: int = MAX_RETRIES) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            # Basic error handling
            error_str = str(exc)
            if "429" in error_str:
                print("Rate limit exceeded. Retrying...")

            if attempt == max_attempts:
                print(f"Final failure: {exc}")
                raise

            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"Attempt {attempt} failed; retrying in {wait}s…")
            time.sleep(wait)


def parse_json_lenient(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = s.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {}


# ==================== OpenAI Wrapper ==================== #
def run_openai_completion(system_prompt: str, user_content: str, json_mode: bool = False) -> Optional[str]:
    def api_call():
        # Truncate to avoid context window errors (approx 30k chars is safe for GPT-4o)
        safe_content = user_content[:30000]

        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_content}
            ],
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"} if json_mode else None
        )
        return response.choices[0].message.content

    return call_with_retries(api_call)


# ==================== SOCIETAL SENTIMENT (Towards Israel) ==================== #
def get_societal_sentiment_with_llm(text_content: str) -> Dict[str, Any]:
    system_prompt = f"""
You are an expert political psychologist.
TASK: Quantify the "Societal Sentiment" directed specifically **TOWARDS ISRAEL** (The State, its citizens, or its companies).

SCOPE:
- Focus on non-governmental actors (Public, Unions, Private Sector, Tourists, Religious Groups).
- **Target:** The sentiment must be ABOUT Israel.
- **Internal vs External:** 
  - If a foreign group (e.g., Turkish Union) hates Israel -> Negative Score.
  - If an Israeli group (e.g., Israeli tourists) expresses patriotism/defense of Israel -> Positive Score.
  - If an Israeli group criticizes their own government (domestic dissent) -> Negative Score.
- EXCLUDE: Purely government-to-government diplomacy or military orders.

SCORING GUIDE (-1.0 to +1.0):
- **-1.0 to -0.8 (Extreme Hostility):** Calls for destruction, violence against Israelis, burning flags, total boycott, "Zionist enemy".
- **-0.7 to -0.3 (Negative/Criticism):** Public protests against policy, canceling vacations in Israel, refusal to buy Israeli goods, anger at leadership.
- **-0.2 to +0.2 (Neutral/Ambivalent):** Business as usual, indifference, or sentiment directed at other targets (e.g., anger at Turkey, not Israel).
- **+0.3 to +0.7 (Supportive/Cooperative):** Increased tourism to Israel, business investment, cultural exchange, "Solidarity with Israel".
- **+0.8 to +1.0 (Unity/Deep Alliance):** Mass rallies supporting Israel, deep strategic friendship, self-sacrifice for the state.

OUTPUT FORMAT (JSON):
{{
  "sentiment_score": float, 
  "sentiment_label": "string",
  "acting_group": "string",
  "description": "string (max 20 words)",
  "evidence": "string (short quote)"
}}

Input Text is in Hebrew. Respond in English.
{_language_clause()}
"""

    try:
        content = run_openai_completion(system_prompt, text_content, json_mode=True)
        if not content:
            return {"sentiment_score": 0.0, "sentiment_label": "Error", "acting_group": "", "description": "",
                    "evidence": ""}

        data = parse_json_lenient(content)

        # Ensure score is float
        score = data.get("sentiment_score", 0.0)
        try:
            score = float(score)
        except ValueError:
            score = 0.0

        return {
            "sentiment_score": score,
            "sentiment_label": data.get("sentiment_label", "None Detected"),
            "acting_group": data.get("acting_group", ""),
            "description": data.get("description", ""),
            "evidence": data.get("evidence", "")
        }

    except Exception as exc:
        return {"sentiment_score": 0.0, "sentiment_label": "Error", "acting_group": "", "description": str(exc),
                "evidence": ""}


# ==================== SUMMARY ==================== #
def get_summary_with_llm(text_content: str) -> str:
    system_prompt = f"Summarize this Hebrew article in English (max 100 words). {_language_clause()}"
    try:
        return (run_openai_completion(system_prompt, text_content, json_mode=False) or "").strip()
    except Exception:
        return ""


# ==================== MAIN ==================== #
OUTPUT_COLUMNS = [
    "NewsID", "Source", "Date", "Title", "summary",
    "societal_sentiment_score",
    "societal_sentiment_label",
    "societal_acting_group",
    "societal_description",
    "societal_evidence",
]


def main():
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {DATA_CSV_PATH}")

    # Load Data
    try:
        df_input = pd.read_csv(DATA_CSV_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_input = pd.read_csv(DATA_CSV_PATH, encoding="utf-8")

    # Prepare Output
    if not OUTPUT_CSV_PATH.exists():
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    df_existing = pd.read_csv(OUTPUT_CSV_PATH, encoding="utf-8-sig") if OUTPUT_CSV_PATH.exists() else pd.DataFrame()
    processed_ids = set(df_existing[NEWS_ID_COLUMN].astype(str)) if not df_existing.empty else set()

    print(f"--- Starting Sentiment Analysis (OpenAI Direct) on {len(df_input)} articles ---")

    for idx, row in df_input.iterrows():
        news_id = str(row.get(NEWS_ID_COLUMN, "")).strip()
        content = str(row.get(CONTENT_COLUMN, "")).strip()

        if not news_id or not content or news_id in processed_ids:
            continue

        print(f"Processing {idx + 1}/{len(df_input)} (ID={news_id})")

        # 1. Get Sentiment Score
        sent_data = get_societal_sentiment_with_llm(content)

        # 2. Get Summary
        summary = get_summary_with_llm(content)

        record = {
            "NewsID": news_id,
            "Source": row.get(SOURCE_COLUMN, ""),
            "Date": row.get(DATE_COLUMN, ""),
            "Title": row.get(TITLE_COLUMN, ""),
            "summary": summary,
            "societal_sentiment_score": sent_data.get("sentiment_score", 0.0),
            "societal_sentiment_label": sent_data.get("sentiment_label", ""),
            "societal_acting_group": sent_data.get("acting_group", ""),
            "societal_description": sent_data.get("description", ""),
            "societal_evidence": sent_data.get("evidence", ""),
        }

        pd.DataFrame([record], columns=OUTPUT_COLUMNS).to_csv(
            OUTPUT_CSV_PATH, mode="a", header=False, index=False, encoding="utf-8-sig"
        )
        processed_ids.add(news_id)

        # Slight pause is polite, though OpenAI tier limits apply
        time.sleep(0.5)

    print(f"Results saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
