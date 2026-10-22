@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0local_model_inventory.ps1" > "%~dp0local_model_inventory.txt" 2>&1
echo inventory complete
type "%~dp0local_model_inventory.txt"
