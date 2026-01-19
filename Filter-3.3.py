import json
import os
import pandas as pd
from pathlib import Path
from openai import OpenAI  # Changed import
from tqdm import tqdm

# ==================== CONFIGURATION ==================== #

# Standard OpenAI Model (e.g., "gpt-4o", "gpt-4-turbo")
OPENAI_MODEL = "gpt-4.1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Update path to your file
DATA_CSV_PATH = Path(r"C:\Users\Soos\PycharmProjects\CAMEO\Secon_filter\Dunya_Tur2.csv")
OUTPUT_CSV_PATH = Path("Dunya2_filter2_results3_OpenAI.csv")

# ==================== CLIENT SETUP ==================== #

if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# Initialize Standard OpenAI Client
client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ==================== BROAD ANALYSIS FUNCTION ==================== #

def analyze_broad_relevance(title, content):
    """
    Analyzes if the article contains ANY meaningful connection (Political, Economic, Social)
    between Israel and Turkey.
    """

    # 1. System Prompt: Multilingual Analyst
    system_prompt = (
        "You are an expert international relations analyst fluent in English, Turkish, and Hebrew. "
        "Your task is to analyze news articles for connections between Israel and Turkey. "
        "IMPORTANT: regardless of the language of the input article (Turkish, English, or Hebrew), "
        "your output JSON and reasoning must ALWAYS be in ENGLISH."
    )

    # 2. Broad User Prompt
    user_prompt = f"""
    Title: {title}
    Content: {content[:15000]}

    --- TASK ---
    Analyze the text above and determine if there is a **connection, interaction, or relationship** involving BOTH 'Turkey' and 'Israel'.

    Use a **BROAD** definition of relevance. If the news falls into ANY of the following categories, mark it as relevant (true):

    1. **Diplomatic & Political:**
       - Official meetings, treaties, or agreements.
       - Tensions, condemnations, or praise between leaders (e.g., Erdogan, Netanyahu, Peres).
       - Appointment or summoning of ambassadors.

    2. **Economic & Business:**
       - Trade deals, energy pipelines, tourism trends.
       - Business investments or corporate cooperation between the two nations.

    3. **Military & Security:**
       - Arms sales (e.g., drones, tanks), joint military exercises, or intelligence sharing.
       - Security cooperation or conflict.

    4. **Social, Public & Cultural:**
       - Public protests in Turkey regarding Israel (or vice versa).
       - Cultural events, TV series controversies, or media disputes.
       - News regarding the Jewish community in Turkey.
       - NGO activities (e.g., Aid flotillas like Mavi Marmara).

    5. **Indirect/Mediation:**
       - Turkey acting as a mediator for Israel (e.g., Israel-Syria talks).
       - Israel's reaction to Turkish foreign policy.

    --- EXCLUSION RULES (Result = false) ---
    - Exclude articles where the two countries are merely listed together in a generic list (e.g., "Tourists visited Greece, Turkey, and Israel").

    --- OUTPUT FORMAT ---
    Respond with this JSON object only:
    {{
      "is_relevant": true/false,
      "reason": "A concise sentence in ENGLISH explaining why it was filtered IN or OUT."
    }}
    """

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,  # Use standard model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("is_relevant", False), result.get("reason", "No reason provided")

    except Exception as e:
        return False, f"LLM Error: {str(e)}"


# ==================== MAIN LOOP ==================== #

def main():
    if not DATA_CSV_PATH.exists():
        print(f"File not found: {DATA_CSV_PATH}")
        return

    # Load Data
    print("Loading Data...")
    # Using utf-8-sig to handle Turkish characters and potential BOM from Excel
    try:
        df = pd.read_csv(DATA_CSV_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_CSV_PATH, encoding="utf-8")  # Fallback

    df = df.fillna("")

    # Strip whitespace from column names to avoid "Date " vs "Date" issues
    df.columns = df.columns.str.strip()

    print(f"Analyzing {len(df)} articles for broad Israel-Turkey connections...")
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        title = str(row.get('Title', ''))
        content = str(row.get('Content', ''))

        is_relevant, reason = analyze_broad_relevance(title, content)

        # FIXED LINE BELOW: Changed row['Data'] to row.get('Date', '')
        results.append({
            "NewsID": row.get('NewsID', index),
            "Date": row.get('Date', ''),
            "Title": title,
            "Is_Relevant": is_relevant,
            "Reason": reason,
            "Content_Snippet": content[:150]
        })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 40)
    print("ANALYSIS COMPLETE")
    print(f"Total Relevant Articles: {results_df['Is_Relevant'].sum()}")
    print(f"Results saved to: {OUTPUT_CSV_PATH}")
    print("=" * 40)

    # Show a few examples
    if not results_df.empty:
        print("\n--- Examples of Reasoning ---")
        # Filter for relevant=True to show meaningful examples
        relevant_df = results_df[results_df['Is_Relevant'] == True]
        if not relevant_df.empty:
            print(relevant_df[['Title', 'Reason']].head(5))
        else:
            print("No relevant articles found.")


if __name__ == "__main__":
    main()
