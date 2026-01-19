import json
import os
import pandas as pd
from pathlib import Path
from openai import OpenAI  # Changed from AzureOpenAI
from tqdm import tqdm

# ==================== CONFIGURATION ==================== #

# Standard OpenAI Configuration
OPENAI_MODEL = "gpt-4.1"  # Use standard model names (e.g., gpt-4o, gpt-4-turbo)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Update path to your file
DATA_CSV_PATH = Path(r"C:\Users\Soos\PycharmProjects\CAMEO\Secon_filter\Dunya_Tur2.csv")
OUTPUT_CSV_PATH = Path("Dunya2_filter2_results_OpenAI.csv")

# ==================== CLIENT SETUP ==================== #

if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# Initialize Standard OpenAI Client
client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ==================== SIMPLE ANALYSIS FUNCTION ==================== #

def check_simple_relevance(title, content):
    """
    Uses a very simple prompt to check for a connection between Israel and Turkey.
    """

    # 1. Simple System Prompt
    system_prompt = (
        "You are an AI assistant and expertise in Turkish. "
        "Analyze the provided news article and determine if it discusses "
        "a relationship, interaction, or event involving BOTH Israel and Turkey. "
        "Return a JSON object with keys: 'is_relevant' (boolean) and 'reason' (string in English)."
    )

    # 2. Simple User Prompt (Just the data)
    user_prompt = f"Title: {title}\n\nContent: {content[:15000]}"

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,  # Use the standard model name defined above
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

    # Clean headers
    df.columns = df.columns.str.strip()

    print(f"Analyzing {len(df)} articles with simple prompt...")
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        title = str(row.get('Title', ''))
        content = str(row.get('Content', ''))

        # Call the SIMPLE function
        is_relevant, reason = check_simple_relevance(title, content)

        results.append({
            "Title_ID": row.get('Title_ID', index),
            "Date": row.get('Date', ''),
            "Title": title,
            "Is_Relevant": is_relevant,
            "Reason": reason,
            "Snippet": content[:100]
        })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 40)
    print("SIMPLE ANALYSIS COMPLETE")
    print(f"Total Relevant Articles: {results_df['Is_Relevant'].sum()}")
    print(f"Results saved to: {OUTPUT_CSV_PATH}")
    print("=" * 40)

    if not results_df.empty:
        print("\n--- Sample Results ---")
        # Handle case where column names might differ slightly, or Is_Relevant is boolean
        relevant_df = results_df[results_df['Is_Relevant'] == True]
        if not relevant_df.empty:
            print(relevant_df[['Title', 'Reason']].head(5))
        else:
            print("No relevant articles found.")


if __name__ == "__main__":
    main()
