#!/bin/bash

echo " Iniciando instalação do ambiente DeepScalp IA..."

# Atualizando o pip
python -m pip install --upgrade pip

# Instalando as bibliotecas necessárias
echo " Instalando dependências (Requests, Pandas, NumPy)..."
pip install requests pandas numpy

echo " Tudo pronto! Agora configure seu arquivo config_client.py e execute o bot."
