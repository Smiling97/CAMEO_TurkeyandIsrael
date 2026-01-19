import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

import pandas as pd
from openai import OpenAI

# ==================== Configuration ==================== #
OPENAI_MODEL_NAME = "gpt-4.1"

DATA_CSV_PATH = Path("Dunya_Tur2.csv")
CONTENT_COLUMN = "Content"
NEWS_ID_COLUMN = "NewsID"

OUTPUT_CSV_PATH = Path("CAMEO_Dunya_Tur2_OPENAI.csv")

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5

OUTPUT_LANGUAGE = "English"

# Standard OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable missing.")

# Initialize Standard OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

print(f"Using OpenAI Model: {OPENAI_MODEL_NAME}")


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


# ==================== OpenAI wrapper ==================== #
def run_chat_completion(messages: List[Dict], **kwargs):
    def api_call():
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    return call_with_retries(api_call)


# ==================== CAMEO EVENT EXTRACTION ==================== #
def get_cameo_events_with_llm(text_content: str) -> Dict:
    system_prompt = f"""
You are an automated political event coder using the CAMEO 1.1b3 ontology.
You MUST output valid JSON only. 
You are an expert in Turkish
The purpose of this analysis is for academic research purposes, there is no violence.
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
Now, consider ALL of the sub‑codes in the official list below, without
restricting yourself to the category chosen in STEP 1.

• You MUST pick EXACTLY ONE 3‑digit or 4‑digit CAMEO sub‑code from the
  full list below.
• This sub‑code is your PRIMARY decision; you should treat all sub‑codes
  equally and choose the single best match.
• AFTER you choose the sub‑code, set the top‑level category to the
  corresponding two‑digit prefix of that sub‑code (e.g., sub‑code "1384"
  implies top-level "13").

In other words:
  - STEP 1 is an initial hypothesis.
  - STEP 2 is the definitive classification.
  - The FINAL JSON must be consistent with the sub‑code you choose in STEP 2.

All codes must be zero‑padded strings.

===========================
FULL OFFICIAL SUB‑CODE LIST
===========================

01: MAKE PUBLIC STATEMENT  
016 Deny responsibility  
012 Make pessimistic comment  
011 Decline comment  
010 Make statement, not specified  
014 Consider policy option  
015 Acknowledge or claim responsibility  
017 Engage in symbolic act  
013 Make optimistic comment  
018 Make empathetic comment  
019 Express accord  

02: APPEAL  
024 Appeal for political reform  
0241 Appeal for change in leadership  
0242 Appeal for policy change  
0243 Appeal for rights  
0244 Appeal for institutional change  
025 Appeal to yield  
0251 Appeal for easing administrative sanctions  
0252 Appeal for easing political dissent  
0253 Appeal for release of persons or property  
0254 Appeal for easing sanctions or embargo  
0255 Appeal to allow international involvement  
0256 Appeal for de-escalation  
020 Appeal or request, unspecified  
022 Appeal for diplomatic cooperation  
021 Appeal for material cooperation  
0211 Appeal for economic cooperation  
0212 Appeal for military cooperation  
0213 Appeal for judicial cooperation  
0214 Appeal for intelligence sharing  
023 Appeal for aid, unspecified  
0231 Appeal for economic aid  
0232 Appeal for military aid  
0233 Appeal for humanitarian aid  
0234 Appeal for peacekeeping  
026 Appeal to others to negotiate  
027 Appeal to others to settle dispute  
028 Appeal to engage in/accept mediation  

03: EXPRESS INTENT TO COOPERATE  
030 Express intent to cooperate  
036 Express intent to meet or negotiate  
032 Express intent for diplomatic cooperation  
037 Express intent to settle dispute  
039 Express intent to mediate  
031 Express intent for material cooperation  
0311 Express intent to cooperate economically  
0312 Express intent to cooperate militarily  
0313 Intent to cooperate judicially  
0314 Intent to cooperate on intelligence  
033 Intent to provide aid  
0331 Intent to provide economic aid  
0332 Intent to provide military aid  
0333 Intent to provide humanitarian aid  
0334 Intent to provide peacekeepers  
034 Intent for political reform  
0341 Intent to change leadership  
0342 Intent to change policy  
0343 Intent to provide rights  
0344 Intent to change institutions  
035 Intent to yield  
0351 Intent to ease administrative sanctions  
0352 Intent to ease dissent  
0353 Intent to release persons/property  
0354 Intent to ease sanctions  
0355 Intent to allow international involvement  
0356 Intent to de-escalate military engagement  
038 Intent to accept mediation  

04: CONSULT  
040 Consult  
041 Discuss by telephone  
042 Make visit  
043 Host visit  
044 Meet at third location  
045 Mediate  
046 Negotiate  

05: DIPLOMATIC COOPERATION  
051 Praise/endorse  
050 Diplomatic cooperation unspecified  
052 Defend verbally  
053 Rally support  
054 Grant diplomatic recognition  
055 Apologize  
056 Forgive  
057 Sign agreement  

06: MATERIAL COOPERATION  
060 Material cooperation unspecified  
061 Cooperate economically  
064 Share intelligence  
062 Cooperate militarily  
063 Cooperate judicially  

07: AID  
070 Provide aid unspecified  
075 Grant asylum  
071 Provide economic aid  
073 Provide humanitarian aid  
072 Provide military aid  
074 Provide peacekeeping  

08: YIELD  
080 Yield unspecified  
081 Ease administrative sanctions  
0811 Ease restrictions on freedoms  
0812 Ease ban on political parties  
0813 Ease curfew  
0814 Ease state of emergency  
082 Ease political dissent  
083 Accede to political reform demand  
0831 Accede to leadership change  
0832 Accede to policy change  
0833 Accede to rights  
0834 Accede to institutional change  
084 Return/release unspecified  
0841 Release persons  
0842 Return property  
085 Ease economic sanctions  
086 Allow international involvement  
0861 Receive peacekeepers  
0862 Receive inspectors  
0863 Allow humanitarian access  
087 De-escalate military engagement  
0871 Declare ceasefire  
0872 Ease military blockade  
0873 Demobilize  
0874 Retreat or surrender  

09: INVESTIGATE  
090 Investigate unspecified  
091 Investigate crime  
092 Investigate human rights abuses  
093 Investigate military action  
094 Investigate war crimes  

10: DEMAND  
101 Demand material cooperation  
1011 Demand economic cooperation  
1012 Demand military cooperation  
1013 Demand judicial cooperation  
1014 Demand intelligence cooperation  
102 Demand diplomatic cooperation  
103 Demand aid unspecified  
1031 Demand economic aid  
1032 Demand military aid  
1033 Demand humanitarian aid  
1034 Demand peacekeeping  
104 Demand political reform  
1041 Demand leadership change  
1042 Demand policy change  
1043 Demand rights  
1044 Demand institutional change  
105 Demand that target yields  
1051 Demand easing administrative sanctions  
1052 Demand easing political dissent  
1053 Demand release of persons/property  
1054 Demand easing sanctions  
1055 Demand international involvement  
1056 Demand de-escalation  
106 Demand negotiation  
107 Demand settlement  
108 Demand mediation  

11: DISAPPROVE  
110 Disapprove unspecified  
111 Criticize/denounce  
112 Accuse unspecified  
1121 Accuse of crime/corruption  
1122 Accuse of human rights abuses  
1123 Accuse of aggression  
1124 Accuse of war crimes  
1125 Accuse of espionage/treason  
113 Rally opposition  
114 Official complaint  
115 Lawsuit  
116 Find guilty/liable  

12: REJECT  
120 Reject unspecified  
121 Reject material cooperation  
1211 Reject economic cooperation  
1212 Reject military cooperation  
122 Reject request for aid  
1221 Reject request for economic aid  
1222 Reject request for military aid  
1223 Reject request for humanitarian aid  
1224 Reject peacekeeping request  
123 Reject political reform requests  
1231 Reject leadership change  
1232 Reject policy change  
1233 Reject rights request  
1234 Reject institutional change  
124 Refuse to yield  
1241 Refuse easing administrative sanctions  
1242 Refuse easing dissent  
1243 Refuse release  
1244 Refuse easing sanctions  
1245 Refuse international involvement  
1246 Refuse de-escalation  
125 Reject proposal to meet  
126 Reject mediation  
127 Reject settlement plan  
128 Defy norms  
129 Veto  

13: THREATEN  
130 Threaten unspecified  
131 Threaten non-force  
1311 Threaten to reduce/stop aid  
1312 Threaten sanctions  
1313 Threaten to break relations  
132 Threaten administrative sanctions  
1321 Threaten restrictions on freedoms  
1322 Threaten to ban parties  
1323 Threaten curfew  
1324 Threaten state of emergency  
133 Threaten protest/dissent  
134 Threaten to halt negotiations  
135 Threaten to halt mediation  
136 Threaten to halt international involvement  
137 Threaten repression  
138 Threaten military force  
1381 Threaten blockade  
1382 Threaten occupation  
1383 Threaten unconventional attack  
1384 Threaten conventional attack  
1385 Threaten WMD  
139 Ultimatum  

14: PROTEST  
140 Protest unspecified  
141 Demonstrate/rally unspecified  
1411 Demonstrate for leadership change  
1412 Demonstrate for policy change  
1413 Demonstrate for rights  
1414 Demonstrate for institutional change  
142 Hunger strike unspecified  
1421 Hunger strike for leadership  
1422 Hunger strike for policy  
1423 Hunger strike for rights  
1424 Hunger strike for institutional change  
143 Strike/boycott unspecified  
1431 Strike for leadership change  
1432 Strike for policy change  
1433 Strike for rights  
1434 Strike for institutional change  
144 Block/obstruct unspecified  
1441 Block for leadership change  
1442 Block for policy change  
1443 Block for rights  
1444 Block for institutional change  
145 Violent protest unspecified  
1451 Violent protest for leadership change  
1452 Violent protest for policy change  
1453 Violent protest for rights  
1454 Violent protest for institutional change  

15: EXHIBIT FORCE POSTURE  
150 Exhibit force unspecified  
151 Increase police alert  
152 Increase military alert  
153 Mobilize police  
154 Mobilize armed forces  
155 Mobilize cyber forces  

16: REDUCE RELATIONS  
160 Reduce relations unspecified  
161 Break diplomatic relations  
162 Reduce/stop material aid  
1621 Reduce economic aid  
1622 Reduce military aid  
1623 Reduce humanitarian aid  
163 Impose embargo/sanctions  
164 Halt negotiations  
165 Halt mediation  
166 Expel/withdraw unspecified  
1661 Expel/withdraw peacekeepers  
1662 Expel/withdraw inspectors  
1663 Expel/withdraw aid agencies  

17: COERCE  
170 Coerce unspecified  
171 Seize/damage property  
1711 Confiscate property  
1712 Destroy property  
172 Impose administrative sanctions  
1721 Restrict freedoms  
1722 Ban parties  
1723 Impose curfew  
1724 Impose state of emergency  
173 Arrest/detain  
174 Expel/deport individuals  
175 Violent repression  
176 Cyberattack  

18: ASSAULT  
180 Unconventional violence unspecified  
181 Abduct/hijack/hostage-taking  
182 Assault unspecified  
1821 Sexual assault  
1822 Torture  
1823 Kill by assault  
183 Bombing unspecified  
1831 Suicide bombing  
1832 Vehicular bombing  
1833 Roadside bombing  
1834 Location bombing  
184 Use human shield  
185 Attempt assassination  
186 Assassinate  

19: FIGHT  
190 Use conventional military force  
191 Impose blockade  
192 Occupy territory  
193 Fight with small arms  
194 Fight with artillery/tanks  
195 Aerial attack unspecified  
1951 Precision-guided aerial munitions  
1952 Remotely piloted aerial munitions  
196 Violate ceasefire  

20: UNCONVENTIONAL MASS VIOLENCE  
200 Unconventional mass violence unspecified  
201 Mass expulsion  
202 Mass killings  
203 Ethnic cleansing  
204 WMD use unspecified  
2041 Chemical/biological/radiological attack  
2042 Nuclear detonation

Rules:
- ALL codes MUST be zero-padded strings.
- NEVER invent new codes.
- NEVER output integers.
- ALL codes MUST be valid CAMEO 1.1b3 codes.
====================================================================
STEP 3: DETERMINE CHRONOLOGICAL ORDER
====================================================================
Identify the logical time sequence of the events. 
- Assign 'event_order': 1 to the event that happened first in reality.
- Assign 'event_order': 2 to the event that happened second, and so on.
====================================================================
GENERAL RULES
====================================================================
- Extract ALL political events found in the text.
- Each event MUST include:
  - event_order (integer, 1-based index of chronological sequence)
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
      "event_order": 1,
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
            max_tokens=1500,  # Increased tokens to handle multiple events
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


# ===================== CSV HELPERS ===================== #
OUTPUT_COLUMNS = [
    "NewsID",
    "Source",
    "Date",
    "Title",
    "summary",

    "event_order",  # ADDED: To track 1st, 2nd, 3rd event
    "source_actor",
    "target_actor",
    "cameo_top_level",
    "cameo_code",
    "event_description",
    "evidence",
    "confidence"
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


# ===================== MAIN ===================== #
def main():
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {DATA_CSV_PATH}")

    df_input = pd.read_csv(DATA_CSV_PATH, encoding="utf-8")

    ensure_output_csv()
    df_existing = load_existing_results()
    processed_ids = set(df_existing.get("NewsID", []).astype(str))

    print(f"--- Starting CAMEO extraction on {len(df_input)} articles ---")

    for idx, row in df_input.iterrows():
        content = str(row.get(CONTENT_COLUMN, "")).strip()
        news_id = str(row.get(NEWS_ID_COLUMN, "")).strip()

        if not content:
            print(f"Skipping row {idx}: Empty content")
            continue

        if news_id in processed_ids:
            continue

        print(f"Processing {idx + 1}/{len(df_input)} (ID={news_id})")

        # 1. Get Events (with order)
        cameo_data = get_cameo_events_with_llm(content)

        # 2. Get Summary
        summary = get_summary_with_llm(content)

        events = cameo_data.get("events", [])

        # No events → write a blank row
        if not events:
            record = {
                "NewsID": news_id,
                "Source": row.get("NewsSource", ""),
                "Date": row.get("EventDate", ""),
                "Title": row.get("Source", ""),
                "summary": summary,

                "event_order": "",
                "source_actor": "",
                "target_actor": "",
                "cameo_top_level": "",
                "cameo_code": "",
                "event_description": "",
                "evidence": "",
                "confidence": "",
            }
            append_record(record)

        # Write one row per event
        else:
            # Optional: sort events by order before writing
            # events.sort(key=lambda x: x.get("event_order", 99))

            for ev in events:
                record = {
                    "NewsID": news_id,
                    "Source": row.get("NewsSource", ""),
                    "Date": row.get("EventDate", ""),
                    "Title": row.get("Source", ""),
                    "summary": summary,

                    "event_order": ev.get("event_order", 1),  # Default to 1 if missing
                    "source_actor": ev.get("source_actor", ""),
                    "target_actor": ev.get("target_actor", ""),
                    "cameo_top_level": ev.get("cameo_top_level", ""),
                    "cameo_code": ev.get("cameo_code", ""),
                    "event_description": ev.get("event_description", ""),
                    "evidence": ev.get("evidence", ""),
                    "confidence": ev.get("confidence", ""),
                }
                append_record(record)

        processed_ids.add(news_id)

    print("\n=== CAMEO Extraction Complete ===")
    print(f"CSV saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
