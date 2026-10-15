@echo off
REM Set your key FIRST in this prompt:  set ANTHROPIC_API_KEY=...   (never paste it in chat)
cd /d "D:\First year\Papers\GeoAgent\revision_ACAGS"
for %%T in (PorFac PermFac) do (
 for %%D in (AB BA) do (
  python llm_orchestrator.py --task %%T --direction %%D
 )
)
python llm_orchestrator.py --analyse_all > orch_results.txt
type orch_results.txt
