# Models and data used in the paper

Companion to *GeoAgent: Reliability-Aware Reservoir Facies–Property Translation with LLM-Orchestrated Abstention and Auditing* (Applied Computing and Geosciences, ms. ACAGS-D-26-00312).

## Checkpoints (release assets)

Four independent Pix2Pix models, one per translation direction, epoch-100 weights (generator and discriminator), trained as described in Section 3.1 of the paper (UNet generator, ngf 64; 70×70 PatchGAN; L1 weight 100; 100 epochs; batch 32; Adam, learning rate 2e-4, beta1 0.5; 256×256 patches). Download them from the **Releases** page of this repository (tag `acags-revision`).

| Asset | Direction | Size (bytes) | SHA-256 |
|---|---|---|---|
| `geoagent_PorFac_AB_ep100.pth` | porosity → facies | 228,812,203 | `27d97f396ae834e905519ab8eeb20f60ce4fc4fdcc1be03fd1cfdcd6fc7ad2ba` |
| `geoagent_PorFac_BA_ep100.pth` | facies → porosity | 228,812,203 | `2bbd3a8657d2ec60ad1c6ac90420bd2288dc5c2c3edf0b3b16c0e268920dcff9` |
| `geoagent_PermFac_AB_ep100.pth` | permeability → facies | 228,812,203 | `d5fd797df24e5ab9951faf91499c1d739fe13f148eb2693ded0436ceae4d4b6b` |
| `geoagent_PermFac_BA_ep100.pth` | facies → permeability | 228,812,203 | `874329fcbe1dcdc3135e824e1e5a6b5434674a17fd04474d58abd69b31e9ffd1` |

Verify a download with `sha256sum <file>` (Linux/macOS) or `certutil -hashfile <file> SHA256` (Windows).

Load a checkpoint:

```python
import torch
from src.geoagent_model import Pix2PixModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Pix2PixModel(device=device)          # 3-channel input, 3-channel output, ngf 64
model.load("geoagent_PorFac_AB_ep100.pth")   # restores generator and discriminator
model.G.eval()
```

## Data

The images are colour renderings of the openly available Groningen static reservoir model (EPOS-NL, DOI 10.24416/UU01-1QH0MW); Section 2 of the paper describes the rendering and patching. The full-field paired image datasets (facies, permeability, porosity and water saturation; training, validation and test sets) are openly available on Zenodo and are not duplicated in this repository:

- Original Datasets: <https://doi.org/10.5281/zenodo.14603410> (CC BY 4.0).
- GeoPix code repository (archived release): <https://doi.org/10.5281/zenodo.14639446>.

The training patches are obtained from the full-field images by the tiling described below (512 x 512 tiles, stride 384, resized to 256 x 256).

| Task | Train patches | Validation patches | Test patches |
|---|---|---|---|
| porosity–facies | 27,135 | 5,805 | 5,835 |
| permeability–facies | 27,120 | 5,805 | 5,835 |

- Patches are 256×256 crops from 512×512 tiles (stride 384), stored as paired PNG images, and numbered sequentially within each split folder (`train/`, `val/`, `test/`).
- The split is made at the level of full-field layer pairs, so that no layer appears in more than one split.
- Property images are rendered with a per-layer colour normalisation, so property errors in the paper are reported in colour-scale units (a fraction of the colour bar), not in physical units.
- The assignment of individual patches to full-field layers and tile positions was not recorded when the dataset was created, so the split cannot be reconstructed layer by layer from the patch files alone.
- The same four checkpoints were used for all experiments in the revised paper (`revision/`).
