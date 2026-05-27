@echo off
REM ============================================================
REM Batch extract 37-marker keypoints from all CSVs in a folder
REM Usage: Edit DATA_DIR below, then double-click or run this .bat
REM ============================================================

REM === EDIT THIS: directory containing raw OptiTrack CSVs ===
set DATA_DIR=C:\Users\zouan\Desktop\CV_Group\Codes\cvslice\data\15\raw_csv

REM === Optional parameters ===
set SKIPROWS=1
set OFFSET=0
set TOTAL_FRAMES=-1

REM === Script location (same folder as this .bat) ===
set SCRIPT_DIR=%~dp0
set EXTRACT_SCRIPT=%SCRIPT_DIR%extract_37_keypoint_from_csv.py

echo ============================================================
echo  Batch Extract 37 Markers
echo  Directory: %DATA_DIR%
echo ============================================================
echo.

set COUNT=0
for %%f in ("%DATA_DIR%\*.csv") do (
    echo %%~nxf | findstr /i /b "extracted" >nul
    if errorlevel 1 (
        set /a COUNT+=1
        echo [Processing] %%~nxf
        python "%EXTRACT_SCRIPT%" -input_csv "%%f" -output_csv "%DATA_DIR%\extracted_%%~nf_37.csv" -skiprows %SKIPROWS% -offset %OFFSET% -total_frames %TOTAL_FRAMES%
        echo.
    )
)

echo ============================================================
echo  Done! Processed files in %DATA_DIR%
echo ============================================================
pause
