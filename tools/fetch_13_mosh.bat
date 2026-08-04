@echo off
REM Download subject 13's MoSh/SMPL results from the lab server to E:\mosh-results\13
REM Server: angus@147.8.93.66 port 23, dir /home/angus/2026_moshpp_for_latest_data
REM You will be asked for the SSH password (twice: once to list, once to copy).

echo === Step 1: what does the server have for subject 13? ===
ssh -p 23 angus@147.8.93.66 "ls /home/angus/2026_moshpp_for_latest_data; echo ---; find /home/angus/2026_moshpp_for_latest_data -iname '*stageii*.pkl' 2>/dev/null | grep -i 13 | head -20; echo ---; find /home/angus/2026_moshpp_for_latest_data -iname '*stageii*.pkl' 2>/dev/null | grep -i 13 | wc -l"

echo.
set /p GO="If the listing above shows 13's files, press Y to download (Y/N): "
if /i not "%GO%"=="Y" goto :eof

if not exist "E:\mosh-results\13" mkdir "E:\mosh-results\13"
echo === Step 2: downloading anything matching *13* ... ===
scp -P 23 -r "angus@147.8.93.66:/home/angus/2026_moshpp_for_latest_data/soma_2026_workspace/runtime_mosh/mosh_results/data/13-*" "E:\mosh-results\13\"

echo.
echo === Done. Local result: ===
dir /s /b "E:\mosh-results\13\*.pkl" | find /c ".pkl"
echo (count of pkl files downloaded)
pause
