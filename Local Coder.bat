@echo off
rem Local Coder - agente de codigo local (llama.cpp + Qwen3.6)
cd /d "%~dp0"
python run.py %*
if errorlevel 1 pause
