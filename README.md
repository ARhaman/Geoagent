# GeoAgent

Reliability-aware agentic framework with LLM orchestration for bidirectional reservoir property translation (facies <-> porosity, facies <-> permeability) on the Groningen gas field model.

Accompanies the manuscript *"GeoAgent: A Reliability-Aware Agentic Framework with LLM Orchestration for Bidirectional Reservoir Property Translation"* (Al-Fakih; under revision, Applied Computing and Geosciences, ACAGS-D-26-00312; the title may change).

## What this repository contains

- **Backbone** – four independent Pix2Pix models, one per direction (porosity->facies, facies->porosity, permeability->facies, facies->permeability): 8-level U-Net generator (ngf 64) and 70x70 PatchGAN, Adam (2e-4, beta1 0.5), lambda_L1 = 100 (taken from Pix2Pix, not tuned), 100 epochs, batch 32.
- **Revised control loop (PERCEIVE, DECIDE, ACT, LEARN, AUDIT)** – test-time augmentation (TTA x8, the full dihedral group); reference-free reliability from TTA consistency and a ridge estimator fitted on validation data; thresholds are quantiles of validation scores (no ground truth at inference); validation-gated fine-tuning with rollback; an LLM orchestrator (tool use) that applies the written policy and writes an audit note.
- **Evaluation** – geologic risk (boundary-tolerant facies accuracy; property MAE in colour-scale units), conventional baselines, within-facies realism, risk-coverage analysis, blind LLM judge and pairwise selector, unseen-orientation test, vertical coherence.
- **Original protocol** – `src/geoagent_agent.py` is the original ReAct loop whose gate used SSIM against the ground truth and whose assessor saw the reference. It is kept for reproducibility of the first submission only and is superseded by `revision/`.

Findings in brief (details in the manuscript; sensitivity of the L1 weight in `revision/results/lambda`): the SSIM gain attributed to the agent is reproduced by fixed TTA x5; fixed TTA x8 is at least as good as every controller; the backbone clearly beats conventional baselines for property->facies and matches a context-aware network on pixelwise error for facies->property; TTA lowers pixelwise error but smooths within-facies variability (the single-pass backbone matches the truth's within-facies distribution; `revision/realism_single.py`); the LLM adds orchestration and auditability but not quality assessment; withholding the 20% least reliable patches lowers delivered risk by 7-20% with no risk guarantee.

## Repository structure

```
src/        model, dataset, training, original agent (see note above)
revision/   scripts, batch files and compact results for the revised analyses
  baselines*.py, realism.py        conventional baselines, within-facies heterogeneity
  gate_*.py, quality_estimator.py  reliability signals, estimator, selective prediction
  ablation_*.py                    static vs TTA vs gated delivery
  gated_learning.py                validation-gated fine-tuning with rollback
  llm_orchestrator.py              LLM orchestrator (tool use); llm_blind_judge.py, llm_pairwise.py
  shift_*.py, ood_signal.py        unseen-orientation test
  vertical_eval.py                 vertical coherence
  lambda_sweep.py                  sensitivity to the L1 weight
  fig4_make.py, make_figures.py    figures
  legend_decode.py, geo_metrics.py colour-legend decoding and geologic-risk measures
  results/                         result files and logs (policy CSVs, JSON summaries, LLM logs)
  RESULTS_SUMMARY.md               every reported number with the script that produced it
notebooks/  original end-to-end notebooks
```

## Data

The Groningen static model is openly available (NAM, 2020; EPOS-NL): <https://doi.org/10.24416/UU01-1QH0MW>. The rendered paired-image datasets are not redistributed; they are available from the corresponding author on reasonable request. Note: rendered property images are normalised per layer, so property errors are in colour-scale units, and the dataset does not record which depth layers belong to which split.

## Checkpoints

Checkpoints (`pix2pix_{AB,BA}_ep0100.pth`) load with `src/geoagent_model.py` (layer names e1..e8, d1..d8). An earlier version of this file used different layer names and could not load them.

## Running the revision scripts

Most scripts take `--code_dir`, `--data_base`, `--ckpt_base` and `--rev_dir`; defaults point to the author's folders, so set them for your machine. The LLM scripts need an Anthropic API key in the environment (`ANTHROPIC_API_KEY`); it is never read from or written to a file.

```
python revision/gate_signals.py ...        # see the header of each script for arguments
```

## License

MIT, see `LICENSE`.

## Contact

Abdulrahman Al-Fakih, Center for Integrative Petroleum Research, King Fahd University of Petroleum and Minerals.
