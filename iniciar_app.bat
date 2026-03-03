@echo off
title Research Copilot – RAG
cd /d C:\CELESTE\QLAB\trabajo
echo Iniciando Research Copilot...
echo La app estara disponible en: http://localhost:8502
echo.
echo Deja esta ventana abierta mientras usas la app.
echo Cierra esta ventana para detener el servidor.
echo.
C:\Users\crist\AppData\Local\Programs\Python\Python312\Scripts\streamlit.exe run app.py --server.port 8502
pause
