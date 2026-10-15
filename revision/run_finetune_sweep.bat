@echo off
REM Runs the full refinement sweep. Safe to restart: existing outputs are skipped.
REM Usage (Anaconda Prompt):  conda activate geoldm_gpu  &&  cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"  &&  run_finetune_sweep.bat
setlocal
set OUT=finetune
if not exist %OUT% mkdir %OUT%
for %%T in (PermFac PorFac) do (
 for %%D in (AB BA) do (
  for %%S in (0 1 2) do (
   for %%B in (0.02 0.05) do (
    for %%M in (hard random) do (
     if not exist %OUT%\%%T_%%D_%%M_s%%S_b%%B.csv (
      echo === %%T %%D %%M seed %%S budget %%B
      python finetune_hard.py --task %%T --direction %%D --mode %%M --budget %%B --seed %%S
     )
    )
   )
   if not exist %OUT%\%%T_%%D_replay_s%%S_b0.05.csv (
    echo === %%T %%D replay seed %%S
    python finetune_hard.py --task %%T --direction %%D --mode replay --budget 0.05 --seed %%S
   )
  )
 )
)
echo ALL DONE
python finetune_analysis.py > finetune_results.txt
type finetune_results.txt
