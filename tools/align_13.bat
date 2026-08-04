@echo off
REM Align subject 13 (same pipeline actor 15 went through):
REM   measure per-view lead (check/ overlay + calibration) -> trim videos+pkl
REM   -> data\smpl\13_aligned (pkl + 30hz) + data\13\13_export_aligned (videos)
REM   -> then copy CSVs + offsets.json so the corrector can open the folder.
REM Requires: python with cv2+numpy. Takes 1-3 hours (re-encodes ~1900 clips).
REM Known gaps (expected warnings): 13-elsdon has only 15/273 videos (export
REM died partway), 13-boss is missing 6 videos - those clips are skipped.

cd /d C:\Users\zouan\Desktop\CV_Group\Codes\cvslice

for %%S in (boss elsdon gallery star travel trove) do (
  echo.
  echo ================= aligning 13-%%S =================
  python data\smpl\15\align_dataset.py ^
    --pkl_dir data\smpl\13\13-%%S ^
    --video_dir data\13\exported_13\13-%%S ^
    --camera_dir data\13\exported_13\13-%%S\calibration ^
    --out_video_dir data\13\13_export_aligned\13-%%S ^
    --out_pkl_dir data\smpl\13_aligned\13-%%S ^
    --jobs 6 --overwrite
  copy /y "data\13\exported_13\13-%%S\*.csv" "data\13\13_export_aligned\13-%%S\" >nul
  copy /y "data\13\exported_13\13-%%S\offsets.json" "data\13\13_export_aligned\13-%%S\" >nul
)

echo.
echo All scenes done. Check the reports (residual_lead should be 0 +/-1):
echo   data\13\13_export_aligned\13-^*\alignment_report.json
pause
