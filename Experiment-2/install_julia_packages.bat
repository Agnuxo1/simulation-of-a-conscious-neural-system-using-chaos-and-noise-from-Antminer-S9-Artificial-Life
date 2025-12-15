@echo off
echo Starting Julia Package Installation for CHIMERA
echo =============================================

echo Opening Julia REPL...
echo Please run the following commands in the Julia REPL:
echo.
echo 1. Press ] to enter package mode
echo 2. Type: add DifferentialEquations Flux CUDA
echo 3. Press Backspace to exit package mode
echo 4. Type: using DifferentialEquations; using Flux; using CUDA
echo 5. Type: CUDA.functional()
echo.
pause

julia.exe

echo Julia session ended.
pause