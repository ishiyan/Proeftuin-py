# C:\Users\ivas\AppData\Local\anaconda3\python.exe
import google.generativeai as genai
import os

#os.environ['SSL_NO_VERIFY'] = 'True'
#os.environ['SSL_VERIFY'] = 'False'

#genai.configure(api_key="", transport="rest")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.")
print(response.text)
