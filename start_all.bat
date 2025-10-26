@echo off
echo ============================================
echo LiveKit Voice Assistant - Starting Services
echo ============================================
echo.

echo [1/3] Starting Token Server...
start "Token Server" cmd /k "python token_server.py"
timeout /t 2 /nobreak >nul

echo [2/3] Starting Voice Agent...
start "Voice Agent" cmd /k "python agent\voice_agent.py"
timeout /t 2 /nobreak >nul

echo [3/3] Starting Web Server...
start "Web Server" cmd /k "cd web && python -m http.server 8002"

echo.
echo ============================================
echo All services started!
echo ============================================
echo.
echo Token Server: http://localhost:5000
echo Web Interface: http://localhost:8002
echo.
echo Make sure LiveKit server is running:
echo   docker run --rm -p 7880:7880 -e LIVEKIT_KEYS="devkey: secret" livekit/livekit-server --dev --node-ip=127.0.0.1
echo.
echo Make sure Ollama is running with llama3:
echo   ollama run llama3:8b
echo.
pause
