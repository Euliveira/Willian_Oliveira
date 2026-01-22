# -*- coding: utf-8 -*-
import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# Importação dos tokens do arquivo config_cliente.py
try:
    from config_cliente import TELEGRAM_TOKEN, TELEGRAM_CHATID
except ImportError:
    print("❌ Erro: Tokens do Telegram não encontrados no config_cliente.py")
    exit()

# --- Configurações Profissionais ---
BASE_URL = "https://api.binance.com"
DB_FILE = "intelligence_core.csv"
MIN_VOL_USDT = 10000000  # Só analisa moedas com > 10M de volume (SMC real)
TREND_THRESHOLD = 15.0   # Moedas que subiram mais de 15% (Tendência Forte)

def enviar_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHATID, "text": mensagem, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e: print(f"Erro Telegram: {e}")

# ----------------------------
# Lógica SMC e Momentum
# ----------------------------
def analisar_smc(df):
    """
    Identifica padrões de Smart Money:
    - Identifica a tendência (Média Móvel Rápida)
    - Procura por exaustão de venda em tendência de alta
    """
    p_atual = df['c'].iloc[-1]
    fechamentos = df['c']
    
    # Média de 20 períodos para confirmar tendência
    sma20 = fechamentos.rolling(20).mean().iloc[-1]
    tendencia_alta = p_atual > sma20
    
    # Cálculo de RSI (Momentum)
    diff = fechamentos.diff()
    g = diff.clip(lower=0).rolling(14).mean()
    p = -diff.clip(upper=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + (g/(p + 1e-10))))
    rsi_val = rsi.iloc[-1]
    
    # Identifica BOS (Break of Structure) simples: Preço atual rompendo a máxima dos últimos 10 candles
    max_recente = fechamentos.iloc[-10:-1].max()
    bos = p_atual > max_recente

    return tendencia_alta, rsi_val, bos

def main():
    enviar_telegram("🕵️ *SMC SCANNER ATIVADO*\nBuscando pegadas do Smart Money em Tendências Fortes...")
    
    while True:
        try:
            # 1. Scanner de Mercado (Busca as moedas mais fortes)
            tickers = requests.get(f"{BASE_URL}/api/v3/ticker/24hr").json()
            
            # Filtro: USDT, Volume Alto e Tendência Forte (>15%)
            oportunidades = [
                t['symbol'] for t in tickers 
                if t['symbol'].endswith('USDT') 
                and float(t['quoteVolume']) > MIN_VOL_USDT 
                and float(t['priceChangePercent']) > TREND_THRESHOLD
            ]

            for s in oportunidades[:5]:
                # 2. Análise Técnica Profunda (5 minutos)
                r = requests.get(f"{BASE_URL}/api/v3/klines", params={"symbol": s, "interval": "5m", "limit": 50}).json()
                df = pd.DataFrame(r, columns=['t','o','h','l','c','v','ct','qv','n','tb','tq','i']).astype(float)
                
                t_alta, rsi_val, bos = analisar_smc(df)
                
                # Regra SMC: Tendência de Alta confirmada + BOS + Pullback de RSI (exaustão vendedora)
                if t_alta and bos and (30 < rsi_val < 50):
                    enviar_telegram(
                        f"💎 *SMC DETECTADO: {s}*\n\n"
                        f"📈 *Estrutura:* BOS (Break of Structure)\n"
                        f"🚀 *Tendência:* Forte Alta (24h)\n"
                        f"📊 *RSI Pullback:* {rsi_val:.2f}\n"
                        f"💡 *Ação:* Instituições acumulando no recuo."
                    )
                    # O monitoramento para aprendizado (IA) continua aqui...
                    time.sleep(2)

            time.sleep(60)
        except Exception as e:
            print(f"Erro: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
