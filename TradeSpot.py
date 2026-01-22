import os
import time
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urlencode
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Chaves de API
try:
    from config_cliente import API_KEY, API_SECRET
except:
    print("Crie o arquivo config_cliente.py com API_KEY e API_SECRET")
    exit()

# --- Configurações Avançadas ---
BASE_URL = "https://api.binance.com"
QUANTIDADE_USDT = 25.0
STOP_LOSS_PCT = 0.99   # -1%
TAKE_PROFIT_PCT = 1.10 # +10% (Alvo inicial, pode ser maior)
BRAIN_FILE = "bot_intelligence.csv" # Onde o bot salva o que aprendeu

# ----------------------------
# Inteligência e Memória
# ----------------------------

def save_to_brain(data):
    """Salva o resultado do trade para a IA aprender no futuro."""
    file_exists = os.path.isfile(BRAIN_FILE)
    df = pd.DataFrame([data])
    df.to_csv(BRAIN_FILE, mode='a', index=False, header=not file_exists)

def get_market_momentum():
    """Escaneia o mercado em busca de moedas com potencial (Volatilidade > 10%)"""
    try:
        r = requests.get(f"{BASE_URL}/api/v3/ticker/24hr")
        tickers = r.json()
        # Filtra moedas USDT que subiram mais de 10% e tem volume alto
        oportunidades = [
            t['symbol'] for t in tickers 
            if t['symbol'].endswith('USDT') and float(t['priceChangePercent']) > 10.0
        ]
        return oportunidades[:10] # Foca nas 10 melhores
    except:
        return ["BTCUSDT", "ETHUSDT", "PEPEUSDT", "DOGEUSDT"]

# ----------------------------
# Funções de API (Privadas e Públicas)
# ----------------------------

def binance_request(method, endpoint, params={}):
    params['timestamp'] = int(time.time() * 1000)
    query = urlencode(params)
    signature = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE_URL}{endpoint}?{query}&signature={signature}"
    headers = {'X-MBX-APIKEY': API_KEY}
    
    if method == 'POST': return requests.post(url, headers=headers).json()
    return requests.get(url, headers=headers).json()

def get_candles(symbol):
    url = f"{BASE_URL}/api/v3/klines"
    params = {'symbol': symbol, 'interval': '15m', 'limit': 100}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','cts','q','n','tb','tq','i'])
    return df[['o','h','l','c','v']].astype(float)

# ----------------------------
# Núcleo de IA Evolutiva
# ----------------------------

def ia_predict_evolution(df, symbol):
    """Usa Random Forest para prever potencial e consulta o 'Brain'."""
    df['returns'] = df['c'].pct_change() * 100
    df = df.dropna()
    
    X = df[['o','h','l','c','v']].values
    y = df['returns'].shift(-1).fillna(0).values # Tenta prever o próximo retorno
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    prediction = model.predict(X[-1].reshape(1, -1))[0]
    
    # Consulta Memória: Se já perdemos dinheiro com esse padrão, diminui a confiança
    if os.path.exists(BRAIN_FILE):
        memory = pd.read_csv(BRAIN_FILE)
        past_fails = memory[(memory['symbol'] == symbol) & (memory['result'] < 0)].shape[0]
        prediction -= (past_fails * 0.1) # Penaliza a previsão
        
    return prediction

# ----------------------------
# Loop de Execução
# ----------------------------

def executar_trade():
    print(f"\n--- Escaneando Mercado: {datetime.now()} ---")
    moedas = get_market_momentum()
    
    for symbol in moedas:
        try:
            df = get_candles(symbol)
            potencial = ia_predict_evolution(df, symbol)
            preco_entrada = df['c'].iloc[-1]

            print(f"Analisando {symbol}: Potencial IA de {potencial:.2f}%")

            if potencial > 1.5: # Só entra se a IA prever mais de 1.5% de alta imediata
                print(f"💎 OPORTUNIDADE DETECTADA em {symbol}!")
                
                # Ordem de Compra
                ordem = binance_request('POST', '/api/v3/order', {
                    'symbol': symbol, 'side': 'BUY', 'type': 'MARKET', 'quoteOrderQty': QUANTIDADE_USDT
                })

                if 'orderId' in ordem:
                    print(f"✅ Comprado! Monitorando Stop Loss (-1%) e Take Profit...")
                    
                    # Loop de Monitoramento da Posição
                    while True:
                        time.sleep(5)
                        ticker = requests.get(f"{BASE_URL}/api/v3/ticker/price", params={'symbol': symbol}).json()
                        preco_atual = float(ticker['price'])
                        lucro = (preco_atual - preco_entrada) / preco_entrada

                        # Lógica de Saída Automática
                        if preco_atual <= (preco_entrada * STOP_LOSS_PCT):
                            binance_request('POST', '/api/v3/order', {
                                'symbol': symbol, 'side': 'SELL', 'type': 'MARKET', 
                                'quantity': ordem['executedQty']
                            })
                            save_to_brain({'symbol': symbol, 'result': -1, 'date': datetime.now()})
                            print(f"🛑 STOP LOSS ACIONADO em {symbol}")
                            break
                        
                        if preco_atual >= (preco_entrada * TAKE_PROFIT_PCT):
                            # Aqui você pode implementar um "Trailing Stop" para buscar os 400%
                            binance_request('POST', '/api/v3/order', {
                                'symbol': symbol, 'side': 'SELL', 'type': 'MARKET', 
                                'quantity': ordem['executedQty']
                            })
                            save_to_brain({'symbol': symbol, 'result': 1, 'date': datetime.now()})
                            print(f"💰 TAKE PROFIT ALCANÇADO em {symbol}!")
                            break
        except Exception as e:
            continue

if __name__ == "__main__":
    while True:
        executar_trade()
        time.sleep(30)
