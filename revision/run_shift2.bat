@echo off
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
if not exist shift\PermFacN\J python shift_render.py
for %%D in (AB BA) do (
  python shift_eval.py --task PorFac  --set PorFacN  --direction %%D
  python shift_eval.py --task PermFac --set PermFacN --direction %%D
)
python shift_summary.py > shift_results2.txt
type shift_results2.txt
