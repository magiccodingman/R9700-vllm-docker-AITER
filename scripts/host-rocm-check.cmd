@echo off
REM Optional Windows shim. The real helper is host-rocm-check.py.
python "%~dp0host-rocm-check.py" %*
