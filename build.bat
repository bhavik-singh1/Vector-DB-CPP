@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Building NebulaDB Engine...
echo ========================================

:: Check for g++
where g++ >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] g++ not found. Please ensure MSYS2/MinGW is in your PATH.
    pause
    exit /b 1
)

if "%1"=="test" (
    echo [1/2] Compiling tests...
    g++ -std=c++17 -O2 -D_WIN32_WINNT=0x0A00 tests/test_algos.cpp src/Algorithms/KDTree.cpp src/Algorithms/HNSW.cpp -o test_runner -Iinclude
    if %ERRORLEVEL% equ 0 (
        echo [2/2] Running tests...
        .\test_runner.exe
    )
    exit /b %ERRORLEVEL%
)

:: Compile
echo [1/2] Compiling sources...
g++ -std=c++17 -O2 -D_WIN32_WINNT=0x0A00 main.cpp src/Algorithms/KDTree.cpp src/Algorithms/HNSW.cpp src/Ollama/OllamaClient.cpp src/Core/VectorDB.cpp src/Core/DocumentDB.cpp src/Core/StorageManager.cpp -o db -Iinclude -lws2_32

if %ERRORLEVEL% equ 0 (
    echo [2/2] Build successful: db.exe
    echo.
    echo To run the server, type: ./db
) else (
    echo [ERROR] Build failed! Check the errors above.
)

echo ========================================
