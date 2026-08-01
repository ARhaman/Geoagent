# GeoAgent

LLM-guided agentic framework for bidirectional reservoir property translation (facies ↔ porosity, facies ↔ permeability) on the Groningen gas field model. GeoAgent wraps a Pix2Pix (U-Net generator + PatchGAN discriminator) backbone in a ReAct-style control loop (REASON → ACT → OBSERVE → ITERATE) that applies escalating test-time augmentation (TTA) and gates acceptance on structural similarity (SSIM), with an LLM vision assessor (claude-sonnet-5) providing a second, semantic quality check.

This repository accompanies the manuscript *"GeoAgent: An LLM-Guided Agentic Framework for Bidirectional Reservoir Property Translation"* (Al-Fakih, submitted).

## Repository structure

```
geoagent/
├── src/
│   ├── geoagent_model.py      # UNetGenerator, PatchGANDiscriminator, Pix2PixModel wrapper
│   ├── geoagent_dataset.py    # PairedGeoDataset loader + DataLoader factory
│   ├── geoagent_train.py      # Pix2Pix training script (2D and 3D-triplet configurations)
│   ├── geoagent_evaluate.py   # SSIM/PSNR/MAE/pixel-accuracy evaluation + acceptance-rate reporting
│   ├── geoagent_agent.py      # GeoAgent ReAct loop: REASON / ACT / OBSERVE / ITERATE, SSIM gate, TTA escalation
│   └── run_geoagent.py        # Quick-start demo of the full pipeline
├── notebooks/
│   ├── GeoAgent_2D_Fixed.ipynb  # End-to-end 2D pipeline with live claude-sonnet-5 QA observer
│   └── GeoAgent_3D_Fixed.ipynb  # End-to-end 3D adjacent-slice-triplet pipeline
├── configs/
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/ARhaman/Geoagent.git
cd Geoagent
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires Python ≥3.9 and a CUDA-capable GPU for practical training times (models were trained on a single NVIDIA RTX 4500 Ada, 25.8 GB). CPU inference is supported but slow.

## Data

The Groningen static model (facies, porosity, permeability grids; 371 wells) is openly available from Nederlandse Aardolie Maatschappij (NAM) via the EPOS-NL research infrastructure:

- Dataset: NAM (2020). *Petrel geological model of the Groningen gas field, the Netherlands* [Dataset]. Utrecht University Yoda. https://doi.org/10.24416/UU01-1QH0MW

This repository does not redistribute the rendered paired-image datasets or trained checkpoints (too large for version control). To reproduce:

1. Obtain the static model above and render co-registered porosity–facies and permeability–facies slice pairs following the tiling scheme in Section 2 of the manuscript (512×556 px pairs, 256×256 px halves after crop).
2. Arrange each direction's data as:
   ```
   <data_root>/
     train/  cube_i*_j*_k*.png
     val/    cube_i*_j*_k*.png
     test/   cube_i*_j*_k*.png
   ```
3. Paired image datasets and trained checkpoints are also available from the corresponding author upon reasonable request.

## Usage

**Train** (one direction at a time; run both `AB` and `BA` per property pair):

```bash
python src/geoagent_train.py \
  --data_root ./data/PorFac_split \
  --direction AB \
  --epochs 100 \
  --batch_size 32 \
  --ngf 64 \
  --save_dir ./checkpoints/PorFac_AB
```

Add `--use_3d` to train the 9-channel adjacent-slice-triplet configuration.

**Evaluate** on the held-out test split:

```bash
python src/geoagent_evaluate.py \
  --data_root ./data/PorFac_split \
  --model_AB ./checkpoints/PorFac_AB/pix2pix_AB_ep0100.pth \
  --model_BA ./checkpoints/PorFac_BA/pix2pix_BA_ep0100.pth \
  --direction both \
  --output_dir ./results/PorFac
```

Reports SSIM, PSNR, MAE, pixel accuracy, acceptance rate, and per-sample agent efficiency; writes a per-image CSV, a JSON summary, and a qualitative comparison grid.

**Run the full agentic loop** (quick-start demo with untrained models, or supply real checkpoints):

```bash
python src/run_geoagent.py --data_root ./data/PorFac_split \
  --model_AB ./checkpoints/PorFac_AB/pix2pix_AB_ep0100.pth \
  --model_BA ./checkpoints/PorFac_BA/pix2pix_BA_ep0100.pth
```

**LLM-active evaluation** (live claude-sonnet-5 vision QA observer, used for the 200-episode agent-behavior results in the manuscript): see `notebooks/GeoAgent_2D_Fixed.ipynb` (2D) and `notebooks/GeoAgent_3D_Fixed.ipynb` (3D). Set your Anthropic API key first:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

If no API key is present, or the API call fails for any reason (auth, network, rate limit), the agent falls back to the documented rule-based SSIM-binned heuristic (`geoagent_agent.py`), so the acceptance gate itself never depends on live API availability. This fallback path is what produced the Table 4c full-agent 3D results (see Supplementary Section S6 of the manuscript).

## Method summary

- **Backbone**: 8-level U-Net generator (Ronneberger et al., 2015) + 70×70 PatchGAN discriminator (Isola et al., 2017), trained with Adam (lr 2×10⁻⁴, β₁ 0.5), λ_L1 = 100, 100 epochs, batch 32, mixed precision.
- **Agent loop**: REASON (infer direction) → ACT (generator + TTA tool) → OBSERVE (SSIM/PSNR scorer + claude-sonnet-5 vision QA) → ITERATE (escalate TTA 5→8→10 up to 3 retries; accept at SSIM ≥ 0.70 or return best attempt flagged unaccepted).
- **Configurations**: 2D single-slice (3-channel input) and 3D adjacent-slice-triplet (9-channel input, `--use_3d`), each at generator capacity `ngf` 64 or 128.
- **Metrics**: SSIM, PSNR, pixel accuracy, KL divergence, Earth mover's distance (Wasserstein-1), experimental semivariograms, and Monte Carlo dropout epistemic uncertainty (see Section 3.5 of the manuscript for definitions).

## Citation

If you use this code, please cite:

```
Al-Fakih, A. (2026). GeoAgent: An LLM-Guided Agentic Framework for
Bidirectional Reservoir Property Translation. [Manuscript submitted for publication].
```

## License

MIT License — see [LICENSE](LICENSE).

## Contact

Abdulrahman Al-Fakih — alfakihabdulrahman2030@gmail.com
Center for Integrative Petroleum Research & Department of Geosciences,
King Fahd University of Petroleum and Minerals, Dhahran 31261, Saudi Arabia
