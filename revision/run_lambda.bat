@echo off
REM Long job (several hours). Restartable. Run in the geoldm_gpu prompt:  run_lambda.bat
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
for %%L in (10 300) do (
 python lambda_sweep.py --task PermFac --direction AB --lam %%L
 python lambda_sweep.py --task PorFac  --direction BA --lam %%L
)
