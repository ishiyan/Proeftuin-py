from openai import OpenAI
from os import getenv

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"))

completion = client.chat.completions.create(
  extra_headers={
    #"HTTP-Referer": $YOUR_SITE_URL, # Optional, for including your app on openrouter.ai rankings.
    #"X-Title": $YOUR_APP_NAME, # Optional. Shows in rankings on openrouter.ai.
  },
  model="meta-llama/llama-3.2-11b-vision-instruct:free",
  messages=[
    {
      "role": "user",
      "content": "what is hugging face?"
    }
  ]
)

print(completion.choices[0].message.content)
