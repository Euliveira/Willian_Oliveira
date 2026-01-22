# -*- coding: utf-8 -*-
import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# =====================================================
# CONFIGURAÇÕES DE PORTFÓLIO
# =====================================================
DB_FILE = "intelligence_core.csv"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
# Foco em moedas voláteis (Meme e Alts)
ATIVOS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PEPEUSDT", "DOGEUSDT", "SHIBUSDT"]

def get_data(symbol, interval, limit=100):
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(BINANCE_URL, params=params, timeout=5)
        df = pd.DataFrame(r.json(), columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","i"])
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        return df
    except: return None

# =====================================================
# CÉREBRO: APRENDIZADO DE ALTA VALORIZAÇÃO
# =====================================================
def aprender_com_o_mercado(symbol, tecnica, rsi_entrada, preco_entrada):
    """
    Monitora o ativo por um período longo para descobrir o potencial real
    de valorização (Busca de 10% a 400%).
    """
    print(f"\n[IA] Iniciando monitoramento de potencial para {symbol}...")
    max_price = preco_entrada
    start_time = time.time()
    
    # Monitora por 10 minutos (em ambiente real seriam horas)
    while time.time() - start_time < 600: 
        ticker = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}).json()
        current_p = float(ticker['price'])
        
        if current_p > max_price:
            max_price = current_p
            
        # Stop Loss de segurança no monitoramento (-1%)
        if current_p < preco_entrada * 0.99:
            break
        time.sleep(2)

    valorizacao_maxima = ((max_price - preco_entrada) / preco_entrada) * 100
    
    # Salva no "Cérebro"
    log = pd.DataFrame([[
        datetime.now(), symbol, tecnica, rsi_entrada, 
        preco_entrada, max_price, valorizacao_maxima
    ]], columns=["data", "symbol", "tecnica", "rsi", "ent", "max", "potencial_pct"])
    
    log.to_csv(DB_FILE, mode='a', index=False, header=not os.path.exists(DB_FILE))
    print(f"[CÉREBRO] {symbol} monitorado. Potencial máximo atingido: {valorizacao_maxima:.2f}%")

def calcular_probabilidade_explosao(tecnica, rsi_atual):
    """Analisa se o padrão atual já gerou altas > 10% no passado."""
    if not os.path.exists(DB_FILE): return 50.0 # Sem dados, 50/50
    try:
        df = pd.read_csv(DB_FILE)
        # Filtra situações similares
        similares = df[(df['tecnica'] == tecnica) & (abs(df['rsi'] - rsi_atual) <= 3)]
        if len(similares) < 3: return 50.0
        
        # Quantos % das vezes subiu mais de 5%?
        sucessos_explosivos = similares[similares['potencial_pct'] >= 5.0].shape[0]
        probabilidade = (sucessos_explosivos / len(similares)) * 100
        return probabilidade
    except: return 50.0

# =====================================================
# ENGINE DE EXECUÇÃO
# =====================================================
def rsi_calc(close, p=9):
    delta = close.diff()
    g = delta.clip(lower=0).rolling(p).mean()
    l = (-delta.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + (g/(l+1e-10))))

def engine():
    print("--- SISTEMA IA PROFISSIONAL ATIVADO ---")
    while True:
        for s in ATIVOS:
            df = get_data(s, "5m", 50)
            if df is None: continue
            
            p_atual = df['c'].iloc[-1]
            rsi = rsi_calc(df['c']).iloc[-1]
            
            # Cálculo de Inteligência de Reforço
            prob_explosao = calcular_probabilidade_explosao("SMC", rsi)
            
            # Lógica de Entrada (Exemplo: Sweep de Liquidez + RSI Baixo)
            if rsi < 30:
                print(f"[{s}] RSI em {rsi:.2f}. Probabilidade histórica de explosão: {prob_explosao:.2f}%")
                
                if prob_explosao > 60: # Filtro Profissional
                    print(f"🚀 SINAL DE ALTA PROBABILIDADE EM {s}!")
                    # Aqui entraria a ordem via URL API
                    aprender_com_o_mercado(s, "SMC", rsi, p_atual)
            
        time.sleep(10)

if __name__ == "__main__":
    engine()
