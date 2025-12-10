import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

import pandas as pd
from openai import AzureOpenAI


# ==================== Configuration ==================== #
AZURE_DEPLOYMENT_NAME = "gpt-4.1"
AZURE_API_VERSION = "2025-04-01-preview"

DATA_CSV_PATH = Path(
    r"C:\Users\Soos\PycharmProjects\CAMEO\First_filter\hurriyet_Eng_filtered_relevant_articles_3.csv"
)
CONTENT_COLUMN = "Content"
NEWS_ID_COLUMN = "NewsID"

OUTPUT_CSV_PATH = Path("CAMEO_hurriyet_Eng_1.csv")
TOPICS_REPORT_PATH = Path("CAMEO_hurriyet_Eng_1.txt")

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5

OUTPUT_LANGUAGE = "English"

AZURE_API_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", )
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_KEY",)

if not AZURE_API_ENDPOINT or not AZURE_API_KEY:
    raise ValueError("Azure OpenAI endpoint/key missing.")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_API_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

print(f"Connected to endpoint: {AZURE_API_ENDPOINT}")
print(f"Using deployment:     {AZURE_DEPLOYMENT_NAME}")


# ==================== Retry wrapper ==================== #
def call_with_retries(fn: Callable, max_attempts: int = MAX_RETRIES) -> any:
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_attempts:
                raise
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"Attempt {attempt} failed ({exc}); retrying in {wait}s…")
            time.sleep(wait)


def _language_clause():
    return "Respond in Turkish." if OUTPUT_LANGUAGE.lower().startswith("turk") else "Respond in English."


# ==================== Azure OpenAI wrapper ==================== #
def run_chat_completion(messages: List[Dict], **kwargs):
    def api_call():
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content
    return call_with_retries(api_call)


# ==================== CAMEO EVENT EXTRACTION ==================== #
def get_cameo_events_with_llm(text_content: str) -> Dict:
    system_prompt = f"""
You are an automated political event coder using the CAMEO 1.1b3 ontology.
You MUST output valid JSON only. (Azure requirement)

====================================================================
STEP 1: Identify the correct TOP‑LEVEL CAMEO CATEGORY (01–20 only).
====================================================================
Choose EXACTLY one of these 20 major categories:

01 Make Statement
02 Appeal
03 Express Intent to Cooperate
04 Consult
05 Diplomatic Cooperation
06 Material Cooperation
07 Provide Aid
08 Yield
09 Investigate
10 Demand
11 Disapprove
12 Reject
13 Threaten
14 Protest
15 Exhibit Military or Police Posture
16 Reduce Relations
17 Coerce
18 Assault
19 Fight
20 Use Unconventional Mass Violence

The top-level category MUST be returned as a zero‑padded TWO‑DIGIT STRING
(e.g., "01", "05", "11", "19").

====================================================================
STEP 2: Select the correct SUB‑CODE inside that category.
====================================================================
Choose a valid 3‑digit or 4‑digit CAMEO sub‑code from the official ontology.
All codes must be zero‑padded strings.

Rules:
- ALL codes MUST be zero-padded strings.
- NEVER invent new codes.
- NEVER output integers.
- ALL codes MUST be valid CAMEO 1.1b3 codes.

====================================================================
GENERAL RULES
====================================================================
- Extract ALL political events found in the text.
- Each event MUST include:
  - source_actor (ISO-3166 + role)
  - target_actor
  - cameo_top_level (01–20)
  - cameo_code (3–4 digit)
  - event_description
  - evidence (direct quotation)
  - confidence (0.0–1.0)
- If no events, output {{"events": []}}.

Output format:

{{
  "events": [
    {{
      "source_actor": "",
      "target_actor": "",
      "cameo_top_level": "",
      "cameo_code": "",
      "event_description": "",
      "evidence": "",
      "confidence": 0.0
    }}
  ]
}}

If NONE → {{"events": []}}


You MUST output valid JSON only. {_language_clause()}
    """

    try:
        content = run_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": "I will output JSON as instructed."},
                {"role": "user", "content": text_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=900,
        )
        return json.loads(content)

    except Exception as exc:
        return {"events": [], "error": str(exc)}


# ===================== OPTIONAL SUMMARY ===================== #
def get_summary_with_llm(text_content: str) -> str:
    system_prompt = f"You are an expert political analyst. Produce a 120-word summary. {_language_clause()}"

    try:
        out = run_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content}
            ],
            temperature=0.3,
            max_tokens=250,
        )
        return out.strip()
    except Exception as exc:
        return f"Error: {exc}"


# ===================== TOPIC EXTRACTION ===================== #
def get_topics_for_single_doc_llm(text_content: str) -> List[str]:
    system_prompt = f"Extract 2–3 very short noun-phrase topics. {_language_clause()}"

    try:
        out = run_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": "I will output JSON topics."},
                {"role": "user", "content": text_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(out).get("topics", [])
    except Exception as exc:
        return [f"Error: {exc}"]


def cluster_topics_with_llm(all_topics_list: List[List[str]]) -> str:
    flat = [t for lst in all_topics_list for t in lst if not str(t).startswith("Error")]
    if not flat:
        return "No topics available."

    prompt = f"Cluster these into 5–7 themes and describe them: {', '.join(flat)}. {_language_clause()}"

    try:
        out = run_chat_completion(
            [
                {"role": "system", "content": "You are a theme clustering engine."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=700,
        )
        return out.strip()
    except Exception as exc:
        return f"Error: {exc}"


# ===================== CSV HELPERS ===================== #
OUTPUT_COLUMNS = [
    "NewsID",
    "Source",
    "Date",
    "Title",
    "summary",

    "source_actor",
    "target_actor",
    "cameo_top_level",
    "cameo_code",
    "event_description",
    "evidence",
    "confidence",

    "document_topics",
    "document_topics_json",
]


def ensure_output_csv():
    if not OUTPUT_CSV_PATH.exists():
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
            OUTPUT_CSV_PATH,
            index=False,
            encoding="utf-8-sig"
        )


def load_existing_results():
    if OUTPUT_CSV_PATH.exists():
        return pd.read_csv(OUTPUT_CSV_PATH, encoding="utf-8-sig")
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def append_record(record):
    pd.DataFrame([record], columns=OUTPUT_COLUMNS).to_csv(
        OUTPUT_CSV_PATH,
        mode="a",
        header=not OUTPUT_CSV_PATH.exists(),
        index=False,
        encoding="utf-8-sig"
    )


def topics_from_results(df):
    topics = []
    col = df.get("document_topics_json", [])

    for item in col.fillna("[]"):
        try:
            topics.append(json.loads(item))
        except:
            topics.append([])
    return topics


# ===================== MAIN ===================== #
def main():
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {DATA_CSV_PATH}")

    df_input = pd.read_csv(DATA_CSV_PATH, encoding="utf-8")

    ensure_output_csv()
    df_existing = load_existing_results()
    processed_ids = set(df_existing.get("NewsID", []).astype(str))

    all_topics = topics_from_results(df_existing)

    print(f"--- Starting CAMEO extraction on {len(df_input)} articles ---")

    for idx, row in df_input.iterrows():
        content = str(row.get(CONTENT_COLUMN, "")).strip()
        news_id = str(row.get(NEWS_ID_COLUMN, "")).strip()

        if not content:
            print(f"Skipping row {idx}: Empty content")
            continue

        if news_id in processed_ids:
            continue

        print(f"Processing {idx+1}/{len(df_input)} (ID={news_id})")

        cameo_data = get_cameo_events_with_llm(content)
        summary = get_summary_with_llm(content)
        doc_topics = get_topics_for_single_doc_llm(content)
        all_topics.append(doc_topics)

        events = cameo_data.get("events", [])

        # No events → write a blank row
        if not events:
            record = {
                "NewsID": news_id,
                "Source": row.get("NewsSource", ""),
                "Date": row.get("EventDate", ""),
                "Title": row.get("Source", ""),
                "summary": summary,

                "source_actor": "",
                "target_actor": "",
                "cameo_top_level": "",
                "cameo_code": "",
                "event_description": "",
                "evidence": "",
                "confidence": "",

                "document_topics": ", ".join(doc_topics),
                "document_topics_json": json.dumps(doc_topics, ensure_ascii=False),
            }
            append_record(record)

        # Write one row per event
        else:
            for ev in events:
                record = {
                    "NewsID": news_id,
                    "Source": row.get("NewsSource", ""),
                    "Date": row.get("EventDate", ""),
                    "Title": row.get("Source", ""),
                    "summary": summary,

                    "source_actor": ev.get("source_actor", ""),
                    "target_actor": ev.get("target_actor", ""),
                    "cameo_top_level": ev.get("cameo_top_level", ""),
                    "cameo_code": ev.get("cameo_code", ""),
                    "event_description": ev.get("event_description", ""),
                    "evidence": ev.get("evidence", ""),
                    "confidence": ev.get("confidence", ""),

                    "document_topics": ", ".join(doc_topics),
                    "document_topics_json": json.dumps(doc_topics, ensure_ascii=False),
                }
                append_record(record)

        processed_ids.add(news_id)

    # Final topic clustering
    final_df = pd.read_csv(OUTPUT_CSV_PATH, encoding="utf-8-sig")
    all_topics = topics_from_results(final_df)

    print("\n--- Clustering topics ---")
    clusters = cluster_topics_with_llm(all_topics)
    TOPICS_REPORT_PATH.write_text(clusters, encoding="utf-8")

    print("\n=== CAMEO Extraction Complete ===")
    print(f"CSV saved to: {OUTPUT_CSV_PATH}")
    print(f"Topic report: {TOPICS_REPORT_PATH}")


if __name__ == "__main__":
    main()
