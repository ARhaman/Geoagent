@echo off
REM Safe to restart: finished runs are skipped.   Usage: conda activate geoldm_gpu && cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS" && run_gated.bat
if not exist gated mkdir gated
for %%T in (PermFac PorFac) do (
 for %%D in (AB BA) do (
  for %%S in (0 1 2) do (
   if not exist gated\%%T_%%D_s%%S.json (
    echo === %%T %%D seed %%S
    python gated_learning.py --task %%T --direction %%D --seed %%S
   )
  )
 )
)
python gated_summary.py > gated_results.txt
type gated_results.txt
