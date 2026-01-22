#!/bin/bash

echo "Iniciando configuração do DeepScalp IA..."

# Verifica se o Python está instalado
if ! command -v python3 &> /dev/null
then
    echo "Python3 não encontrado. Por favor, instale o Python antes de continuar."
    exit
fi

# Instalação das bibliotecas
echo "Instalando dependências (Requests, Pandas, NumPy)..."
pip install requests pandas numpy

echo "Instalação concluída com sucesso!"
echo "Lembre-se de configurar suas chaves no arquivo config_client.py antes de rodar o bot."
