@echo off
echo ==========================================
echo TAMLELAN V1.1 - PyInstaller Build Script
echo ==========================================

echo [1/3] Cleaning old build caches...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist scribe.spec del /q scribe.spec

echo.
echo [2/3] Compiling scribe.py into standalone executable...
:: --noconsole hides the CMD window
:: --onefile packages everything into a single .exe
:: service_account.json is NOT bundled - place it next to scribe.exe after building
:: models\ (diarization ONNX files) is NOT bundled either - downloaded on first
::   run next to scribe.exe, or can be dropped in manually for an offline install
:: --collect-all sherpa_onnx: sherpa_onnx ships native DLLs alongside its Python
::   extension that PyInstaller's onefile analysis misses by default -- without
::   this flag the exe builds fine but crashes at runtime on import
python -m PyInstaller --noconsole --onefile --collect-all sherpa_onnx scribe.py

echo.
echo [3/3] Verifying build...
if exist dist\scribe.exe (
    echo ==========================================
    echo SUCCESS: scribe.exe has been generated!
    echo You can find it inside the 'dist' folder.
    echo ==========================================
) else (
    echo ==========================================
    echo ERROR: Compilation failed. Check the console output above.
    echo ==========================================
)
pause