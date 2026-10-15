@echo off
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
python baselines2.py --task PorFac  --direction BA
python baselines2.py --task PermFac --direction BA
python realism.py --task PorFac
python realism.py --task PermFac
