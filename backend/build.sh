#!/usr/bin/env bash
# ============================================================
# Build Script para o Backend no Render
# ============================================================
# O Render executa este arquivo como buildCommand quando configurado
# manualmente. Instala dependências com suporte a índice customizado
# para emergentintegrations
# ============================================================
set -o errexit

echo "==> Atualizando pip"
pip install --upgrade pip

echo "==> Instalando dependencias do requirements.txt (com índice customizado)"
pip install -r requirements.txt --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/

echo "==> Build concluido com sucesso"
