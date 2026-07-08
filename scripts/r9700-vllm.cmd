@echo off
REM Optional Windows shim. The real cross-platform helper is r9700-vllm.py.
python "%~dp0r9700-vllm.py" %*
