import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()  # 👈 this was missing in your version

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

response = client.responses.create(
  model="gpt-4o-mini",
  input="write a haiku about ai",
  store=True,
)

print(response.output_text);
