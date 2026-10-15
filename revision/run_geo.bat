@echo off
REM Runs geo_metrics for all 4 directions on val and test. Skips finished outputs, safe to restart.
REM (geoldm_gpu)  cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"  &&  run_geo.bat
setlocal
for %%T in (PorFac PermFac) do (
 for %%D in (AB BA) do (
  for %%S in (val test) do (
   if not exist geo2\%%T_%%D_%%S.csv (
    echo === %%T %%D %%S
    python geo_metrics.py --task %%T --direction %%D --split %%S
   )
  )
 )
)
echo ALL DONE
