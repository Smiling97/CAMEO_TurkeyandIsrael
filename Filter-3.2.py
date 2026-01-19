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
OUTPUT_CSV_PATH = Path("Dunya2_filter2_results2_OpenAI.csv")

# ==================== CLIENT SETUP ==================== #

if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# Initialize Standard OpenAI Client
client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ==================== CAMEO ANALYSIS FUNCTION ==================== #

def check_cameo_relevance(title, content):
    """
    Analyzes if the article contains a CAMEO event between Israel and Turkey.
    Forces output in English even if text is Turkish.
    """

    # 1. System Prompt: Sets the Persona + Language Constraint
    system_prompt = (
        "You are an expert political event coder specializing in the CAMEO and expertise in Turkish"
        "(Conflict and Mediation Event Observations) framework. "
        "You must determine if a valid interaction exists between two specific state actors. "
        "IMPORTANT: Regardless of the language of the input article (Turkish, English, or Hebrew), "
        "your output JSON and reasoning must ALWAYS be in ENGLISH."
    )

    # 2. Detailed User Prompt: Defines CAMEO rules for Israel <-> Turkey
    user_prompt = f"""
    Title: {title}
    Content: {content[:15000]}

    --- TASK ---
    Analyze the text above and determine if it describes a CAMEO event involving BOTH 'Turkey' (TUR) and 'Israel' (ISR).

    To be 'relevant' (true), the text must describe an action taken by Turkey affecting Israel, OR an action taken by Israel affecting Turkey.

    Use these 4 CAMEO 'Quad' categories to decide:
    1. Verbal Cooperation: (e.g., Turkey praises Israel, Israel expresses regret to Turkey, Diplomats meet, Agree to negotiate).
    2. Material Cooperation: (e.g., Israel delivers drones to Turkey, Trade agreements, Joint military drills, Providing aid).
    3. Verbal Conflict: (e.g., Erdogan criticizes Peres, Israel condemns Turkish TV series, Ambassadors summoned for protest).
    4. Material Conflict: (e.g., Expelling diplomats, Canceling military exercises, Seizing ships, Military engagement).

    --- EXCLUSION RULES (Result = false) ---
    - Ignore articles where both countries are just mentioned but do not interact (e.g., "US officials visited Israel and Turkey").
    - Ignore articles about the Israel-Palestine conflict unless Turkey explicitly reacts or intervenes.

    --- OUTPUT FORMAT ---
    Respond with this JSON object only:
    {{
      "is_relevant": true/false,
      "reason": "Identify the specific CAMEO action in ENGLISH (e.g., 'Verbal Conflict: Erdogan criticized Israel at Davos')."
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
    try:
        df = pd.read_csv(DATA_CSV_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_CSV_PATH, encoding="utf-8")

    df = df.fillna("")

    # Strip whitespace from headers to avoid "Date " issues
    df.columns = df.columns.str.strip()

    print(f"Analyzing {len(df)} articles for CAMEO events...")
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        title = str(row.get('Title', ''))
        content = str(row.get('Content', ''))

        # Call the new CAMEO function
        is_relevant, reason = check_cameo_relevance(title, content)

        # Safe extraction of Date using .get()
        results.append({
            "NewsID": row.get('NewsID', index),
            "Date": row.get('Date', ''),
            "Title": title,
            "Is_Relevant": is_relevant,
            "Reason_English": reason,
            "Content_Snippet": content[:150]
        })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 40)
    print("CAMEO ANALYSIS COMPLETE")
    print(f"Total Relevant Events: {results_df['Is_Relevant'].sum()}")
    print(f"Results saved to: {OUTPUT_CSV_PATH}")
    print("=" * 40)

    if not results_df.empty:
        print("\n--- Sample CAMEO Results ---")
        # Ensure we filter boolean correctly
        relevant_rows = results_df[results_df['Is_Relevant'] == True]
        if not relevant_rows.empty:
            print(relevant_rows[['Title', 'Reason_English']].head(5))
        else:
            print("No relevant events found.")


if __name__ == "__main__":
    main()
