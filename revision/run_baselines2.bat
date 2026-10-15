@echo off
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
python baselines2.py --task PorFac  --direction AB
python baselines2.py --task PorFac  --direction BA
python baselines2.py --task PermFac --direction AB
python baselines2.py --task PermFac --direction BA
python baselines_analysis.py --sub geo3b
