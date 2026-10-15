@echo off
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
for %%T in (PorFac PermFac) do (
 for %%D in (AB BA) do (
  if not exist geo5\%%T_%%D_test.csv python ablation_risk.py --task %%T --direction %%D
 )
)
python ablation_table.py > ablation_results.txt
type ablation_results.txt
