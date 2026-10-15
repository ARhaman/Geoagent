@echo off
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
(
python vertical_eval.py --task PorFac
python vertical_eval.py --task PermFac
) > vertical_results.txt 2>&1
type vertical_results.txt
