@echo off
echo Stopping old bot instances...
taskkill /F /FI "WINDOWTITLE eq bot*" /IM python.exe 2>nul
timeout /t 3 /nobreak >nul

echo Starting bot...
cd /d "D:\Claude Cowork\telegram_faceswap_bot"
start "bot" /B python -u bot.py >> bot.log 2>&1

timeout /t 8 /nobreak >nul
echo.
echo === Recent log ===
powershell -command "Get-Content bot.log -Tail 10"
echo.
echo Bot started! Press any key to exit.
pause >nul
