# Módulo Especial de Consultoria na Área de Dados com Agentes de IA
# Projeto Prático Para Consultoria na Área de Dados com Agentes de IA
# Deploy de App Para Day Trade Analytics em Tempo Real com Agentes de IA, OpenAI e AWS

# Imports
import re
import os
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from phi.agent import Agent
# MUDANÇA 1: Importar OpenAIChat ao invés de Groq
from phi.model.openai import OpenAIChat 
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import requests_cache
import requests
from requests import Session

# Carrega o arquivo de variáveis de ambiente
load_dotenv()

# MUDANÇA 2: Carregar a chave da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

########## Analytics (Mantido igual) ##########

import requests_cache

@st.cache_data
def ins_extrai_dados(ticker, period="6mo"):
    # Cria uma sessão que salva os dados em um arquivo .sqlite por 24 horas
    session = requests_cache.CachedSession('yfinance.cache', expire_after=86400)
    session.headers.update({'User-Agent': 'Mozilla/5.0...'}) # Mesmo user-agent de cima

    stock = yf.Ticker(ticker, session=session)
    hist = stock.history(period=period)
    
    hist.reset_index(inplace=True)
    return hist
    
def ins_plot_stock_price(hist, ticker):
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} Preços das Ações (Últimos 6 Meses)", markers=True)
    st.plotly_chart(fig)

def ins_plot_candlestick(hist, ticker):
    fig = go.Figure(
        data=[go.Candlestick(x=hist['Date'],
                             open=hist['Open'],
                             high=hist['High'],
                             low=hist['Low'],
                             close=hist['Close'])]
    )
    fig.update_layout(title=f"{ticker} Candlestick Chart (Últimos 6 Meses)")
    st.plotly_chart(fig)

def ins_plot_media_movel(hist, ticker):
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
    fig = px.line(hist, 
                  x='Date', 
                  y=['Close', 'SMA_20', 'EMA_20'],
                  title=f"{ticker} Médias Móveis (Últimos 6 Meses)",
                  labels={'value': 'Price (USD)', 'Date': 'Date'})
    st.plotly_chart(fig)

def ins_plot_volume(hist, ticker):
    fig = px.bar(hist, 
                 x='Date', 
                 y='Volume', 
                 title=f"{ticker} Trading Volume (Últimos 6 Meses)")
    st.plotly_chart(fig)

########## Agentes de IA (Atualizado para OpenAI) ##########

# MUDANÇA 3: Seleção do Modelo GPT
# "gpt-4o" é excelente para português e raciocínio complexo.
# Alternativa mais barata: "gpt-4o-mini"
MODEL_ID = "gpt-4o" 

# Agentes de IA 
ins_agente_web_search = Agent(
    name="Insider Agente Web Search",
    role="Fazer busca na web",
    model=OpenAIChat(id=MODEL_ID), # Uso do OpenAIChat
    tools=[DuckDuckGo()],
    instructions=["Sempre inclua as fontes", "Responda sempre em Português do Brasil"],
    show_tool_calls=True, 
    markdown=True
)

ins_agente_financeiro = Agent(
    name="Insider Agente Financeiro",
    model=OpenAIChat(id=MODEL_ID), # Uso do OpenAIChat
    tools=[YFinanceTools(stock_price=True,
                         analyst_recommendations=True,
                         stock_fundamentals=True,
                         company_news=True)],
    instructions=["Use tabelas para mostrar os dados", "Responda sempre em Português do Brasil"],
    show_tool_calls=True, 
    markdown=True
)

multi_ai_agent = Agent(
    team=[ins_agente_web_search, ins_agente_financeiro],
    model=OpenAIChat(id=MODEL_ID), # Uso do OpenAIChat
    instructions=[
        "Sempre inclua as fontes", 
        "Use tabelas para mostrar os dados", 
        "Combine as informações financeiras com as notícias da busca web",
        "Responda sempre em Português do Brasil com tom profissional de consultor"
    ],
    show_tool_calls=True, 
    markdown=True
)

########## App Web ##########

st.set_page_config(page_title="Data Insiders", page_icon=":100:", layout="wide")

st.sidebar.title("Instruções")
st.sidebar.markdown("""
### Como Utilizar a App:

- Insira o símbolo do ticker da ação desejada no campo central.
- Clique no botão **Analisar** para obter a análise em tempo real com visualizações e insights gerados por IA.

### Exemplos de tickers válidos:
- MSFT (Microsoft)
- TSLA (Tesla)
- PBR (Petrobras - ADR)
- VALE (Vale - ADR)

### Finalidade da App:
Este aplicativo realiza análises avançadas de preços de ações em tempo real utilizando Agentes de IA com modelo GPT-4o através da OpenAI para apoio a estratégias de Day Trade.
""")

if st.sidebar.button("Suporte"):
    st.sidebar.write("No caso de dúvidas envie e-mail para: suporte@datainsiders.com.br")

st.title(":100: Data Insiders")
# st.image("logo_Insiders.png") # Comentei caso não tenha a imagem localmente
st.text("Agentes de IA desenvolvida Robson Brandão")

st.header("Day Trade Analytics em Tempo Real com Agentes de IA (OpenAI)")

ticker = st.text_input("Digite o Código (símbolo do ticker):").upper()

if st.button("Analisar"):
    if ticker:
        with st.spinner("Buscando os Dados em Tempo Real e Consultando o GPT-4o..."):
            
            hist = ins_extrai_dados(ticker)
            
            st.subheader("Análise Gerada Por IA")
            
            # Prompt reforçando o idioma
            ai_response = multi_ai_agent.run(f"Resumir a recomendação do analista e compartilhar as últimas notícias para {ticker}. Responda em Português do Brasil.")

            clean_response = re.sub(r"(Running:[\s\S]*?\n\n)|(^transfer_task_to_finance_ai_agent.*\n?)","", ai_response.content, flags=re.MULTILINE).strip()

            st.markdown(clean_response)

            st.subheader("Visualização dos Dados")
            ins_plot_stock_price(hist, ticker)
            ins_plot_candlestick(hist, ticker)
            ins_plot_media_movel(hist, ticker)
            ins_plot_volume(hist, ticker)
    else:
        st.error("Ticker inválido. Insira um símbolo de ação válido.")
