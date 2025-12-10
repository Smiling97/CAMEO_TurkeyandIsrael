import json
import os
from pathlib import Path
import pandas as pd
from openai import AzureOpenAI


# ==================== CONFIG ==================== #
AZURE_DEPLOYMENT_MINI = "gpt-4.1"
AZURE_API_VERSION = "2025-04-01-preview"

DATA_CSV_PATH = Path(
    r"C:\Users\Soos\PycharmProjects\CAMEO\First_filter\hurriyet_Eng_filtered_relevant_articles.csv"
)
CONTENT_COLUMN = "Content"
NEWS_ID_COLUMN = "NewsID"

KEYWORD_FILTER_OUTPUT = Path("hurriyet_Eng_keyword_filtered_articles_2.csv")
FINAL_FILTER_OUTPUT = Path("hurriyet_Eng_filtered_relevant_articles_2.csv")
DEBUG_OUTPUT = Path("hurriyet_filter_debug_2.csv")

AZURE_API_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT",)
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_KEY", )

if not AZURE_API_ENDPOINT or not AZURE_API_KEY:
    raise ValueError("Azure OpenAI endpoint/key missing.")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_API_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

print("Connected to Azure.")


# ================================================================
# STEP 1 — EXPANDED MULTILINGUAL KEYWORD FILTER
# ================================================================
def cheap_keyword_filter(text: str) -> bool:
    if not text:
        return False

    t = text.lower()

    # ENGLISH KEYWORDS
    israel_en = ["israel", "israeli", "jerusalem", "gaza", "idf",
                 "netanyahu", "herzog", "sharon"]
    turkey_en = ["turkey", "turkish", "türkiye", "ankara", "istanbul",
                 "erdogan", "erdoğan", "akp"]

    # TURKISH
    israel_tr = ["israil", "israilli", "kudüs", "gazze"]
    turkey_tr = ["türkiye", "türk", "ankara", "istanbul", "erdoğan"]

    # HEBREW
    israel_he = ["ישראל", "ישראלי", "עזה", "ירושלים", "צה\"ל"]
    turkey_he = ["טורקיה", "טורקי", "איסטנבול", "ארדואן"]

    # ADD POLITICAL → to catch high‑level relations
    political_terms = [
        "prime minister", "president", "foreign minister", "diplomat",
        "cabinet", "government", "minister",
        "başbakan", "cumhurbaşkanı",           # Turkish
        "ראש הממשלה", "שר החוץ", "משרד החוץ"  # Hebrew
    ]

    has_israel = any(k in t for k in israel_en + israel_tr + israel_he)
    has_turkey = any(k in t for k in turkey_en + turkey_tr + turkey_he)
    has_political = any(k in t for k in political_terms)

    # Rules:
    if has_israel and has_turkey:
        return True

    # One side + political words (important for diplomacy coverage)
    if has_political and (has_israel or has_turkey):
        return True

    return False


# ================================================================
# STEP 2 — MINI LLM FILTER (CHEAP)
# ================================================================
def llm_relevance_filter(text: str) -> bool:
    """
    Returns True if article is specifically about Israel interacting with Turkey.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a multilingual relevance classifier. "
                "You MUST output valid JSON only. "
                "You understand English, Hebrew, and Turkish.\n\n"
                "Your task is to determine whether an article is specifically about "
                "Israel interacting with Turkey, or Turkey interacting with Israel.\n\n"
                "Definition of 'interaction':\n"
                "- diplomacy, negotiations, agreements\n"
                "- political or military cooperation\n"
                "- conflict, disputes, sanctions, threats\n"
                "- trade, economic relations, joint initiatives\n"
                "- official statements by one state explicitly about the other\n"
                "- actions by leaders, ministers, embassies, diplomats, or institutions "
                "of one country directly concerning the other country\n\n"
                "NOT considered interaction (must be classified as not relevant):\n"
                "- incidental mentions of Turkey or Israel\n"
                "- author/official titles like 'Israel's ambassador to Turkey'\n"
                "- articles about Gaza, Hamas, the UN, or other topics without Israel–Turkey interaction\n"
                "- travel, geography, ethnicity, or background references\n"
                "- historical context that includes either country without bilateral engagement\n\n"
                "When in doubt, answer false.\n\n"
                "Output ONLY JSON in the form: {\"relevant\": true} or {\"relevant\": false}"
            )
        },
        {
            "role": "user",
            "content": (
                "Determine if this article is about Israel interacting with Turkey, "
                "or Turkey interacting with Israel.\n"
                "Respond ONLY in json: {\"relevant\": true/false}\n\n"
                f"Article:\n{text}"
            )
        }
    ]

    # Your model call here:
    # response = client.chat.completions.create( ... )
    # Then parse response.json()["relevant"]


    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_MINI,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("relevant", False)

    except Exception as e:
        print("LLM filter err:", e)
        return False


# ================================================================
# MAIN FILTERING PIPELINE (NOW SAVES EVERYTHING)
# ================================================================
def main():
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {DATA_CSV_PATH}")

    df = pd.read_csv(DATA_CSV_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} articles.\n")

    keyword_filtered_rows = []
    final_filtered_rows = []
    debug_records = []     # Full dataset with yes/no columns

    for idx, row in df.iterrows():
        content = str(row.get(CONTENT_COLUMN, "")).strip()
        news_id = row.get(NEWS_ID_COLUMN)

        print(f"--- Article {idx+1}/{len(df)} (ID={news_id}) ---")

        # STEP 1
        keyword_pass = cheap_keyword_filter(content)
        print("Keyword filter:", keyword_pass)

        # STEP 2
        llm_pass = False
        if keyword_pass:
            llm_pass = llm_relevance_filter(content)
            print("LLM filter:", llm_pass)
        else:
            print("Skipping LLM filter (keyword failed).")

        # RECORD FOR DEBUG CSV
        debug_records.append({
            **row.to_dict(),
            "keyword_pass": "yes" if keyword_pass else "no",
            "llm_pass": "yes" if llm_pass else "no",
        })

        # Save matches
        if keyword_pass:
            keyword_filtered_rows.append(row)

        if keyword_pass and llm_pass:
            final_filtered_rows.append(row)

    # SAVE DEBUG (all articles)
    df_debug = pd.DataFrame(debug_records)
    df_debug.to_csv(DEBUG_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved full debug log to: {DEBUG_OUTPUT}")

    # STEP 1 OUTPUT
    if keyword_filtered_rows:
        pd.DataFrame(keyword_filtered_rows).to_csv(KEYWORD_FILTER_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"Saved {len(keyword_filtered_rows)} keyword-filtered articles to: {KEYWORD_FILTER_OUTPUT}")
    else:
        print("No keyword matches.")

    # STEP 2 OUTPUT
    if final_filtered_rows:
        pd.DataFrame(final_filtered_rows).to_csv(FINAL_FILTER_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"Saved {len(final_filtered_rows)} LLM-confirmed relevant articles to: {FINAL_FILTER_OUTPUT}")
    else:
        print("No LLM-confirmed relevant articles.")


if __name__ == "__main__":
    main()
