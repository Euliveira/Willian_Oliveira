#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bot Bybit - Monitor + SVR online
- Registra sinais (CSV) e logs
- NÃO abre ordens (modo ALARME)
"""

import os
import time
import csv
import logging
import requests
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Importa chaves - tenta os dois nomes comuns (config_cliente.py ou config_client.py)
try:
    from config_cliente import API_KEY, API_SECRET
except Exception:
    try:
        from config_client import API_KEY, API_SECRET
    except Exception:
        API_KEY = ""
        API_SECRET = ""

# scikit-learn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Configurações
# ----------------------------
API_BASE = "https://api.bybit.com"
ASSETS_CACHE_FILE = "bybit_symbols_cache.txt"
CSV_FILE = "bybit_signals.csv"
LOG_FILE = "bybit_trader.log"

CANDLE_LIMIT = 200      # quantidade de candles para treinar
INTERVAL = "60"         # intervalo em minutos para endpoint Bybit (60 == 1h). Você pode trocar.
SLEEP_BETWEEN_CYCLES = 8  # segundos entre ciclos completos
TRAIN_MIN_ROWS = 50     # linhas mínimas para treinar o modelo

# logging
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------
# Utilitários
# ----------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_request_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[requests.Response]:
    """Requisição GET segura com tratamento mínimo de erros."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.RequestException as e:
        logging.warning(f"Requisição GET falhou para {url} params={params}: {e}")
        return None

def registrar_trade_csv(trade_data: Dict[str, Any], csv_file: str = CSV_FILE):
    """Registra o sinal em CSV (append)."""
    file_exists = os.path.isfile(csv_file)
    fieldnames = ["timestamp","symbol","side","entry_price","qty","sl","tp","prediction_pct","status"]
    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_data)
    except Exception as e:
        logging.error(f"Erro ao gravar CSV: {e}")

# ----------------------------
# Bybit - símbolos e candles
# ----------------------------
def get_all_bybit_symbols(category: str = "linear", use_cache: bool = True) -> List[str]:
    """
    Retorna lista de símbolos (categoria: 'linear' para USDT perp).
    Opcionalmente usa cache para evitar várias chamadas seguidas.
    """
    # se existir cache recente, use
    if use_cache and os.path.exists(ASSETS_CACHE_FILE):
        try:
            mtime = os.path.getmtime(ASSETS_CACHE_FILE)
            # se o cache for menor que 3600s (1 hora), use
            if time.time() - mtime < 3600:
                with open(ASSETS_CACHE_FILE, "r", encoding="utf-8") as f:
                    syms = [l.strip() for l in f if l.strip()]
                    if syms:
                        return syms
        except Exception:
            pass

    url = f"{API_BASE}/v5/market/instruments-info"
    symbols = []
    cursor = None

    while True:
        params = {"category": category, "limit": 500}
        if cursor:
            params["cursor"] = cursor

        resp = safe_request_get(url, params=params)
        if not resp:
            break

        try:
            data = resp.json()
        except Exception as e:
            logging.error(f"Resposta inválida ao buscar símbolos: {e}")
            break

        if data.get("retCode") != 0:
            logging.error(f"Erro API ao buscar símbolos: {data}")
            break

        result = data.get("result", {})
        inst_list = result.get("list", [])
        for inst in inst_list:
            sym = inst.get("symbol")
            if sym:
                symbols.append(sym)

        cursor = result.get("nextPageCursor")
        if not cursor:
            break

    # salvar cache
    try:
        with open(ASSETS_CACHE_FILE, "w", encoding="utf-8") as f:
            for s in symbols:
                f.write(s + "\n")
    except Exception:
        pass

    return symbols

def get_bybit_candles(symbol: str, interval: str = INTERVAL, limit: int = CANDLE_LIMIT) -> Optional[pd.DataFrame]:
    """
    Pega candles OHLCV da Bybit (v5 market/kline).
    Retorna DataFrame com colunas: timestamp, open, high, low, close, volume, turnover
    Observação: a API retorna lista com formato [open_time, open, high, low, close, volume, ...]
    """
    url = f"{API_BASE}/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}

    resp = safe_request_get(url, params=params)
    if not resp:
        return None

    try:
        data = resp.json()
    except Exception as e:
        logging.error(f"Erro ao decodificar JSON candles {symbol}: {e}")
        return None

    if data.get("retCode") != 0:
        logging.warning(f"Bybit retornou erro candles {symbol}: {data}")
        return None

    rows = data.get("result", {}).get("list", [])
    if not rows:
        return None

    # A resposta costuma vir mais recente -> mais antiga; garantir ordem cronológica
    try:
        # cada item: [open_time, open, high, low, close, volume, turnover]
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        # Converter tipos
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["turnover"] = df["turnover"].astype(float)
        # Ordenar cronologicamente do mais antigo ao mais recente
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"Erro ao construir DataFrame candles {symbol}: {e}")
        return None

# ----------------------------
# Exibição completa do ativo
# ----------------------------
def exibir_dados_completos(df: pd.DataFrame, symbol: str):
    """Mostra na tela todos os dados possíveis do ativo (último candle)."""
    try:
        ultima = df.iloc[-1]
        ts = int(ultima["timestamp"])
        ts_dt = datetime.fromtimestamp(ts / 1000.0)
        print("\n" + "=" * 80)
        print(f"SYMBOL: {symbol} | TIMESTAMP: {ts_dt} ({ts})")
        print("=" * 80)
        print(f"open:     {ultima['open']}")
        print(f"high:     {ultima['high']}")
        print(f"low:      {ultima['low']}")
        print(f"close:    {ultima['close']}")
        print(f"volume:   {ultima['volume']}")
        print(f"turnover: {ultima['turnover']}")
        # info de janela (últimos candles)
        print("-" * 80)
        last5 = df.tail(5)[["timestamp","open","high","low","close","volume"]].copy()
        last5["timestamp"] = last5["timestamp"].apply(lambda x: datetime.fromtimestamp(int(x)/1000.0))
        print("Últimas 5 velas (mais antigas → mais recentes):")
        print(last5.to_string(index=False))
        print("=" * 80)
    except Exception as e:
        logging.error(f"Erro ao exibir dados completos para {symbol}: {e}")

# ----------------------------
# Análise técnica simples
# ----------------------------
def analisar_tendencia(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], str]:
    """Retorna preço atual, variação do último candle (%) e tendência textual."""
    try:
        if df.shape[0] < 2:
            return None, None, "Dados insuficientes"
        preco_atual = float(df["close"].iloc[-1])
        preco_anterior = float(df["close"].iloc[-2])
        var_pct = (preco_atual - preco_anterior) / preco_anterior * 100.0
        if var_pct > 0:
            trend = "Alta 📈 (compra)"
        elif var_pct < 0:
            trend = "Baixa 📉 (venda)"
        else:
            trend = "Estável ➖"
        return preco_atual, var_pct, trend
    except Exception as e:
        logging.error(f"Erro analisar_tendencia: {e}")
        return None, None, "Erro"

# ----------------------------
# Machine Learning: Treino / Previsão (SVR)
# ----------------------------
def treinar_modelo_svr(df: pd.DataFrame) -> Tuple[Optional[SVR], Optional[StandardScaler], Optional[StandardScaler], Optional[List[str]]]:
    """
    Treina um modelo SVR para prever a variação percentual do próximo candle.
    Retorna: model, scaler_X, scaler_y, feature_columns
    Observação: usamos DataFrames/colunas para evitar warnings de nomes de features.
    """
    try:
        # preparar df: criar variação %
        df_local = df.copy()
        df_local["pct_change"] = df_local["close"].pct_change() * 100.0
        df_local = df_local.dropna().reset_index(drop=True)

        if df_local.shape[0] < TRAIN_MIN_ROWS:
            return None, None, None, None

        # Features escolhidas
        feature_cols = ["open", "high", "low", "close", "volume", "turnover"]
        X_df = df_local[feature_cols].astype(float)
        y_arr = df_local["pct_change"].astype(float).values.reshape(-1, 1)

        # Escalonadores (fit com DataFrame para manter nomes)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # scaler_X aceita DataFrame; para evitar o warning, faremos fit_transform em DataFrame e manteremos col names
        X_scaled = scaler_X.fit_transform(X_df)  # retorna ndarray mas scaler_fitted contém feature_names_in_
        # Para evitar o warning do sklearn quando transformarmos usando um DataFrame mais tarde,
        # guardamos feature_cols e usaremos DataFrame ao transformar previsões.
        y_scaled = scaler_y.fit_transform(y_arr).ravel()

        # Treinar SVR
        model = SVR(kernel="rbf", C=10, gamma="scale")
        model.fit(X_scaled, y_scaled)

        return model, scaler_X, scaler_y, feature_cols

    except Exception as e:
        logging.error(f"Erro ao treinar modelo SVR: {e}")
        return None, None, None, None

def prever_proxima_variacao(df: pd.DataFrame, model: SVR, scaler_X: StandardScaler, scaler_y: StandardScaler, feature_cols: List[str]) -> Optional[float]:
    """
    Recebe DataFrame, modelo treinado e scalers. Retorna previsão da variação % do próximo candle.
    Para evitar aviso 'X does not have valid feature names', vamos construir um DataFrame
    com as mesmas colunas antes de transformar.
    """
    try:
        ultima_features = df[feature_cols].iloc[-1:].astype(float)
        # transformar usando scaler_X - para evitar warning, passamos DataFrame -> scaler_X.transform aceita ndarray;
        # scikit-learn 1.2+ pode reclamar se feature names diferentes; garantir mesma ordem de colunas é suficiente.
        X_scaled = scaler_X.transform(ultima_features)  # aceita ndarray com mesma ordem

        pred_scaled = model.predict(X_scaled)  # retorna 1d array
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]  # variação % prevista
        return float(pred)
    except Exception as e:
        logging.error(f"Erro ao prever próxima variação: {e}")
        return None

# ----------------------------
# Lógica de geração de sinal
# ----------------------------
def gerar_sinal_por_previsao(pred_pct: float, min_buy_pct: float = 0.5, min_sell_pct: float = -0.5) -> Optional[str]:
    """
    Regras de sinal baseadas na previsão:
    - Compra se previsão >= min_buy_pct (ex: 0.5%)
    - Venda se previsão <= min_sell_pct (ex: -0.5%)
    - Senao, None
    """
    try:
        if pred_pct is None:
            return None
        if pred_pct >= min_buy_pct:
            return "Buy"
        if pred_pct <= min_sell_pct:
            return "Sell"
        return None
    except Exception as e:
        logging.error(f"Erro em gerar_sinal_por_previsao: {e}")
        return None

# ----------------------------
# MAIN
# ----------------------------
def main():
    print(f"--- Iniciando Bot Bybit (Modo Alarme) {now_str()} ---")
    logging.info("Bot iniciado (modo alarme).")

    # 1) Carregar símbolos (linear -> perp USDT)
    symbols = get_all_bybit_symbols("linear")
    if not symbols:
        print("❌ Não foi possível carregar símbolos da Bybit. Verifique conexão.")
        return

    print(f"🔎 Símbolos carregados: {len(symbols)} (ex.: {symbols[:8]})")
    logging.info(f"{len(symbols)} símbolos carregados.")

    # laço infinito (cada ciclo roda por todos os símbolos)
    while True:
        ciclo_inicio = time.time()
        for symbol in symbols:
            try:
                df = get_bybit_candles(symbol, interval=INTERVAL, limit=CANDLE_LIMIT)
                if df is None or df.shape[0] < TRAIN_MIN_ROWS:
                    # pular se sem dados suficientes
                    logging.debug(f"Dados insuficientes para {symbol}. Linhas: {None if df is None else df.shape[0]}")
                    continue

                # Exibir dados completos do ativo na tela
                exibir_dados_completos(df, symbol)

                # Analisar tendência simples
                preco_atual, var_pct, trend = analisar_tendencia(df)
                print(f"Resumo: Preço atual={preco_atual} | Última variação={var_pct:.4f}% | Tendência={trend}")

                # Treinar modelo (SVR) com os candles atuais
                model, scaler_X, scaler_y, feature_cols = treinar_modelo_svr(df)
                if model is None:
                    logging.debug(f"Modelo não treinado para {symbol} (poucos dados).")
                    print(f"IA: dados insuficientes para treinar modelo de {symbol}.")
                    continue

                # Prever próxima variação %
                pred_pct = prever_proxima_variacao(df, model, scaler_X, scaler_y, feature_cols)
                if pred_pct is None:
                    print("IA: erro na previsão.")
                    continue

                print(f"🔮 Previsão IA (próx. variação %): {pred_pct:.4f}%")

                # Gerar sinal com regras simples (ajustáveis)
                signal = gerar_sinal_por_previsao(pred_pct, min_buy_pct=0.5, min_sell_pct=-0.5)
                if signal:
                    print(f"📢 SINAL: {signal} para {symbol} (prev {pred_pct:.4f}%)")
                    logging.info(f"SINAL {signal} | {symbol} | prev_pct={pred_pct:.4f}")

                    # calcular SL/TP simples com ATR (pode usar calcular_atr se quiser)
                    # Para manter simples: definimos placeholders
                    sl = None
                    tp = None
                    qty = 0.0

                    trade_data = {
                        "timestamp": now_str(),
                        "symbol": symbol,
                        "side": signal,
                        "entry_price": f"{preco_atual:.8f}" if preco_atual else "0",
                        "qty": f"{qty:.8f}",
                        "sl": f"{sl if sl else 0}",
                        "tp": f"{tp if tp else 0}",
                        "prediction_pct": f"{pred_pct:.6f}",
                        "status": "SIGNAL"
                    }
                    registrar_trade_csv(trade_data)
                else:
                    print("Sem sinal forte pelo modelo (neutro).")

                print("-" * 80)

            except Exception as e:
                logging.error(f"Erro no processamento do símbolo {symbol}: {e}")
                # não quebra todo o loop se um símbolo falhar
                continue

        ciclo_fim = time.time()
        dur = ciclo_fim - ciclo_inicio
        print(f"\nCiclo completo finalizado em {dur:.1f}s. Aguardando {SLEEP_BETWEEN_CYCLES}s para próxima rodada...\n")
        time.sleep(SLEEP_BETWEEN_CYCLES)


# ----------------------------
# Execução
# ----------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
    except Exception as e:
        logging.critical(f"Erro fatal no main: {e}")
        print("Erro crítico. Veja o log para detalhes.")