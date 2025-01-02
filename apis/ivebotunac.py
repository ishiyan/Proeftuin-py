# pip install pypdf
# pip install yfinance
# pip install ta
# pip install -U langchain-openai
# pip install chromadb

import os

import textwrap
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
#from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
#from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
#from openai import OpenAI

#import google.generativeai as genai
#genai.configure(api_key="AIzaSyBG-9zVx3nflBoemGyin-u_Fq3zi22M2mw", transport="rest")
#model = genai.GenerativeModel("gemini-1.5-flash")

# Load environment variables
#load_dotenv()
#openai_api_key = os.getenv('OPENAI_API_KEY')

# Prepare a Retrieval-Augmented Generation (RAG) system
# to enhance AI's ability to process and analyze finantial
# reports and market data efficiently.

# Load and process the PDF document
#loader = PyPDFLoader('FYY24_Q3_Consolidated_Financial_Statements.pdf')
#pages = loader.load_and_split()
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#texts = text_splitter.split_documents(pages)

# Create vector database
embeddings = OpenAIEmbeddings(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)
#db = Chroma.from_documents(texts, embeddings)
vectorstore = Chroma("langchain_store", embeddings)

# Setup RAG system
llm = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
  temperature=0,
)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())

# Create a function to calculate various technical indicators,
# providing crutial insights into stock performance and market trends.

# Calculate technical indicators for a given ticker.
def calculate_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period='1y')

    # Simple Moving Average (SMA)
    history['SMA50'] = ta.trend.sma_indicator(history['Close'], window=50)
    history['SMA200'] = ta.trend.sma_indicator(history['Close'], window=200)

    # Relative Strength Index (RSI)
    history['RSI14'] = ta.momentum.srsi(history['Close'], window=14)

    # MACD
    macd = ta.trend.MACD(history['Close'])
    history['MACD'] = macd.macd()
    history['MACD_Signal'] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(history['Close'], window=20, window_dec=2)
    history['BB_Upper'] = bollinger.bollinger_hband()
    history['BB_Lower'] = bollinger.bollinger_lband()

    current_price = history['Close'].iloc[-1]
    sma50 = history['SMA50'].iloc[-1]
    sma200 = history['SMA200'].iloc[-1]
    rsi14 = history['RSI14'].iloc[-1]
    macd_value = history['MACD'].iloc[-1]
    macd_signal = history['MACD_Signal'].iloc[-1]
    bb_upper = history['BB_Upper'].iloc[-1]
    bb_lower = history['BB_Lower'].iloc[-1]

    return f'''Technical Indicators for {ticker}:
    Current Price: ${current_price:.2f}
    SMA50: ${sma50:.2f}
    SMA200: ${sma200:.2f}
    RSI14: {rsi14:.2f}
    MACD: {macd_value:.2f}
    MACD Signal: {macd_signal:.2f}
    Bollinger Bands:
        Upper: ${bb_lower:.2f}
        Lower: ${bb_upper:.2f}'''

# Create a function to retrieve current stock information,
# enabling the AI agent to access up-to-date market data for analysis.

# Retrieve current stock information for a given ticker.
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)

    # Get historical data
    history = stock.history(period='1d')
    last_price = history['Close'].iloc[-1]

    # Get additional information
    info = stock.info
    company_name = info.get('longName', ticker)
    market_cap = info.get('marketCap', 'N/A')
    pe_ratio = info.get('trailingPE', 'N/A')
    dividend_yield = info.get('dividendYield', 'N/A')

    # Format dividend yield if available
    if dividend_yield != 'N/A':
        dividend_yield = f'{dividend_yield:.2%}'

    # Get average volume
    avg_volume = info.get('averageVolume', 'N/A')

    if market_cap != 'N/A':
        market_cap = f'${market_cap:,.0f}'

    return f'''
    Company: {company_name} ({ticker})
    Last price: ${last_price:.2f}
    Market Cap: {market_cap}
    Sector: {info['sector']}
    P/E Ratio: {pe_ratio}
    Dividend Yield: {dividend_yield}
    Average Volume: {avg_volume:,}
    '''

# Create tools and ReAct agent
# Configure the AI tools, including Company Report QA, Stock Info, and Technical Indicators.
# Set up the ReAct agent to integrate these tools for a comprehensive analysis of financial data.

# Create tools for the agent
tools_for_agent = [
    Tool(
        name='Company Report QA',
        func=qa.run,
        description='''Use this to extract specific financial information or performance metrics from the company's latest financial report. Input should be a question about the report content.'''
        ),
    Tool(
        name='Stock Info',
        func=get_stock_info,
        description='''Retrieve current market data for financial analysis, including stock price, market cap, P/E ratio, and dividend yield. Input should be a only stock ticker symbol (e.g. AAPL for Apple).'''
        ),
    Tool(
        name='Technical Indicators',
        func=calculate_technical_indicators,
        description='''Calculate technical indicators including SMA, RSI, MACD, and Bollinger Bands. Input should be a stock ticker symbol (e.g. AAPL for Apple).'''
        )
]

# Set up the ReActagent
react_prompt = hub.pull('hwchase17/react')
agent = create_react_agent(llm, tools_for_agent, react_prompt)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools_for_agent, verbose=True)

# Define the query for analysis
# Formulate specific queries to guide the AI agent in performing targeted stock market analysis,
# ensuringrelevant and actionable insights.

# Define the company name to analyze
company_name = 'Apple'

# Construct the query for analysis
query = f'''Analyze the company {company_name} with a focus on the second half of 2024. Please provide the following information:

1. From the latest financial report for the second half of 2024:
   a) What is the total revenue, and how does it compare to the same period last year?
   b) What is the net income, and how has it changed year-over-year?
   c) What is the gross margin percentage?

2. Current stock information:
   a) What is the current stock price?
   b) What is the current market capitalization?
   c) What is the price-to-earnings(P/E) ratio?

3. Technical analysis:
   a) Provide a brief overview of the stock's technical indicators.

4. Based on the financial report data, current market information, and technical analysis:
   a) Provide an investment recommendation (buy, hold, or sell) with a detailed explanation of yourreasoning.
   b) Suggest a potential price targetfor the next 6-12 months, if available

    Remember to use only the provided tools'''

# Execute agent and show results
# Run the AI agent with the defined query and present the analysis results,
# offering valuable insights for investing strategies.

# Execute the agent with the query
response = agent_executor.invoke({'input': query})

#Print the agent's response
print('Agent response:')
wrapped_response = textwrap.fill(response['output'], width=180)
print(wrapped_response)
