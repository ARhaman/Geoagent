@echo off
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
if not exist shift\PermFac\J python shift_render.py
for %%T in (PorFac PermFac) do (
 for %%D in (AB BA) do (
  python shift_eval.py --task %%T --direction %%D
 )
)
python shift_summary.py > shift_results.txt
type shift_results.txt
