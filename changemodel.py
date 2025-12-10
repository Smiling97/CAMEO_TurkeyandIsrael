#if you are using Gemini or OpenAI model directly, you can change the header to
from openai import OpenAI
client = OpenAI(api_key=api_key) # insert your openai key
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}]
)

# or if you use Gemini
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-2.5-pro")
chat = model.start_chat(history=[])

# I used OpenAN model Via another Azure OpenAI, the setting is a bit different with connect directly with OpenAI or Gemini, but other setting should be the same.

