@echo off
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
(
for %%T in (PorFac PermFac) do (
 for %%D in (AB BA) do (
  python ood_signal.py --task %%T --direction %%D
 )
)
) > ood_results.txt 2>&1
type ood_results.txt
