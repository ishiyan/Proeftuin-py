# C:\Users\ivas\AppData\Local\anaconda3\python.exe
import google.generativeai as genai
import os

proxy = 'http://localhost:3128'
#proxy = 'http://sg.proxy.mid.dom:80'
#os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
#os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
#os.environ['SSL_NO_VERIFY'] = 'True'
#os.environ['SSL_VERIFY'] = 'False'

# conda config --set ssl_verify False

# $env:SSL_NO_VERIFY = 'True'
# $env:SSL_VERIFY = 'False'
# dir env:

#$env:HTTP_PROXY = 'http://localhost:3128'
#$env:HTTPS_PROXY = 'http://localhost:3128'
#C:\Users\ivas\AppData\Local\anaconda3\Scripts\pip.exe
#C:\Users\ivas\AppData\Local\anaconda3\python.exe


#genai.configure(api_key="", transport="rest")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.")
print(response.text)
