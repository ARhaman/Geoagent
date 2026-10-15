# GeoAgent revision (ACAGS-D-26-00312): single source of truth for all new results

All numbers below come from scripts in this folder (script name in brackets). "Risk": property->facies (AB) = 1 - boundary-tolerant facies accuracy; facies->property (BA) = MAE in colour-bar units (see Section 0). Lower is better. Test set = 5,835 patches per task (4,925-4,985 valid after masks). Intervals = paired bootstrap 95% CI.

## 0. Data facts discovered (affect how results must be worded)
- Rendered property images are normalised PER LAYER: permeability bar = that layer's own log10 min-max (rms 0.065-0.072 in bar fraction, layers 81/121/150/201); porosity bar = layer percentile-type range (spans 0.048-0.139 ... 0.07-0.226 across layers; exact rule not identified). Legend decoded from one screenshot is therefore NOT universal. "v/v" and "log10 mD" errors are in legend-scale units; honest unit = fraction of colour bar (porosity MAE 0.0107 of a 0.107 bar = 10%; permeability 0.2535 of a 3.0 bar = 8.5%). [diagnose_data.py + Real-layers fits]
- Grid vs render layer index: layer N <-> grid depth index k = N-2 (facies match 100%, layer 150).
- Porosity: 20.6% of in-field pixels saturate at the bar limits (MAE excludes them; including them: single 0.0095, TTAx8 0.0089). Permeability: 3.0% saturated, 4.0% off-bar. Neither property is discontinuous (lag-1 roughness 0.021 / 0.044); permeability has ~2.3x more short-scale variability (lag-4 roughness 0.428 vs 0.186). [diagnose_data.py]
- Adjacent depth layers agree on facies 59.8% (lag 1) -> 48.8% (lag 20); laterally 82.6% (lag 1). Copying the neighbouring layer therefore scores ~0.60 << GAN pixel accuracy 0.86: a layer-level split cannot be exploited by neighbour memorisation. Which depth layers are in the test split is NOT recorded in the dataset.
- SSIM implemented globally (not windowed); earlier gate used ground truth and earlier LLM assessor saw ground truth + SSIM (non-blind): must be disclosed.

## 1. Backbone vs conventional baselines [baselines.py, baselines2.py, baselines_analysis.py]
Same test patches, TTAx8 GAN.
| Direction | Metric | GAN | Pixelwise baseline | Context-aware MLP | GAN - MLP |
|---|---|---|---|---|---|
| Por->Fac | bdry-tolerant acc | 0.9431 | 0.8116 (Bayes) | 0.8866 | +0.0565 [+0.0542,+0.0589] |
| Por->Fac | facies-proportion err | 0.0544 | 0.1176 | 0.0746 | -0.0202 |
| Perm->Fac | bdry-tolerant acc | 0.8597 | 0.6348 | 0.7181 | +0.1416 [+0.1373,+0.1460] |
| Perm->Fac | proportion err | 0.0767 | 0.1537 | 0.1340 | -0.0573 |
| Fac->Por | MAE | 0.0107 | 0.0143 (class mean) | 0.0110 | -0.0004 [-0.0005,-0.0003] (tie) |
| Fac->Perm | MAE | 0.2535 | 0.2906 | 0.2492 | +0.0043 (MLP slightly better) |

## 2. Heterogeneity realism, facies->property [realism.py]  (2,000 test patches)
Within-facies std, truth / GAN / MLP:
- Fac->Por: sand 0.0163 / 0.0147 / 0.0110; shaly sand 0.0201 / 0.0176 / 0.0107; shale 0.0100 / 0.0122 / 0.0017.
- Fac->Perm: sand 0.302 / 0.300 / 0.169; shaly sand 0.346 / 0.153 / 0.140; shale 0.416 / 0.249 / 0.169.
Semivariogram ratio to truth (lags 1-16): GAN 0.45-0.64 (Por sand/shaly), 0.37-1.08 (Por shale); MLP 0.2-0.4 and ~0. Fac->Perm GAN 0.51-0.64 sand, 0.13-0.24 shaly sand, 0.21-0.59 shale; MLP 0.12-0.21, 0.03-0.13, 0.03-0.10.
Wasserstein-1: GAN closer in Por sand (0.0037 vs 0.0051) and shaly sand (0.0042 vs 0.0074), MLP closer in Por shale; Fac->Perm about equal except shale (GAN 0.169 vs MLP 0.222). Both under-dispersed in Fac->Perm shaly sand/shale; GAN over-dispersed in Por shale.
Claim: GAN reproduces much more within-facies heterogeneity than a smooth model; it ties on pixelwise MAE.

## 3. Agent vs static pipeline [ablation_risk.py, ablation_table.py]
| Direction | A single-pass | B +TTAx8 | C gate 90/80/70% delivered risk | abstained-patch risk (70%) |
|---|---|---|---|---|
| Por->Fac | 0.0965 | 0.0569 (-0.0395 [-0.0410,-0.0380]) | 0.0567/0.0579/0.0576 | 0.0553 (no discrimination) |
| Perm->Fac | 0.1688 | 0.1403 (-0.0285) | 0.1390/0.1327/0.1232 | 0.1756 |
| Fac->Por | 0.0111 | 0.0107 (-0.0004) | 0.0103/0.0101/0.0100 | 0.0121 |
| Fac->Perm | 0.2915 | 0.2535 (-0.0380) | 0.2487/0.2477/0.2482 | 0.2641 |
Risk-control target (0.85 x val mean): met on test only for Por->Fac (at 39.5% coverage), missed for Perm->Fac (0.1155 vs 0.1133), not reachable on val for BA. => no guarantee claimed; report risk-coverage trade-off only.

## 4. Second reliability signal: input-side OOD distance [ood_signal.py] (exploratory)
Delivered risk at 70% coverage (consistency / OOD / combined): Por->Fac 0.0576/0.0520/0.0540; Perm->Fac 0.1263/0.1290/0.1228; Fac->Por 0.0100/0.0108/0.0103; Fac->Perm 0.2477/0.2648/0.2586. Helps property->facies, wrong-signed for facies->property. Flags 96-100% of my rendered depth controls too -> cannot be validated as a geological shift detector.

## 5. Validation-gated learning with rollback [gated_learning.py, gated_summary.py] (3 seeds/direction)
Accepted: Por->Fac 0/3, Perm->Fac 0/3, Fac->Por 3/3, Fac->Perm 2/3. Ungated candidate mean dRisk (all patches): +0.0067 (Por->Fac), +0.0044 (Perm->Fac; hard patches -0.0249), -0.0002, -0.0018. Gated deployed dRisk: 0, 0, -0.0002, -0.0014. No deployed update worse than base beyond 2% tolerance. Rule is conservative: Perm->Fac gains on hard patches were rolled back (check split noisy). Earlier fine-tune sweep (finetune_results.txt): Perm->Fac hard-patch bad-rate 54.3% -> 35.9% (budget 0.05), random control 43.4%, replay-only 43.6%.

## 6. LLM roles
- Judge (blind) and pairwise action selector: no demonstrated value (earlier results in llm\ and llm_pair\).
- Orchestrator/auditor [llm_orchestrator.py, claude-sonnet-5, 242 decisions, 4 directions, half near-threshold]: policy fidelity 100%, tool-call correctness 100%, repeat-consistency 100% (120 pairs), audit-report numbers traceable 100% (44/44). A fixed rule also scores 100%: value = natural-language interface + non-hallucinating audit trail, not accuracy.

## 7. Orientation shift test [shift_render.py, shift_eval.py, shift_summary.py, ood_signal.py]  (same NAM field, depth = seen, I/J cross-sections = unseen; cells rendered square)
Valid controls: Fac->Perm (K 0.237 vs 0.254), Por->Fac only with absolute-scale render (K 0.0559 vs 0.0569). Partial: Perm->Fac (K 0.223 vs 0.140), Fac->Por (K 0.0150 vs 0.0107).
- Por->Fac: risk 0.056 -> 0.141 (I) / 0.135 (J); gate abstains 1% / 0%; accepted 0.1455 vs abstained 0.1101 (gate fails). Sensitive to input intensity scaling (per-slice p2/p98 render: control 0.184, I/J ~0.62).
- Perm->Fac: K 0.223 -> I 0.279 / J 0.312; accepted 0.112/0.121 vs abstained 0.568/0.579 (gate works, rho -0.91/-0.88).
- Fac->Perm: no degradation (0.259/0.255); accepted 0.231 vs abstained 0.376.
- Fac->Por: no degradation (0.0113); gate false alarm (abstains 77-82%).
=> consistency gate is not a universal shift detector. Original S5 cross-dataset collapse (SSIM 0.138/0.006) should be reported with per-layer normalisation as a likely cause (hypothesis). The independent S5 dataset was not found in the connected folders.

## 8. Vertical coherence [vertical_eval.py] (depth layers 100-132, 10 tiles; layer split membership unknown)
Por->Fac: facies agreement at lag 1/2/4/8 true 0.639/0.583/0.577/0.572, predicted 0.636/0.566/0.556/0.555; accuracy on these tiles 0.950; neighbour-copy 0.639. Perm->Fac render only partially validated (accuracy 0.563): do not cite.

## 9. Still to do before resubmission
Re-express property metrics in colour-bar fractions; fix Figure 4 (label rows, correct bars/mean); GenAI disclosure; repo fix (nb_model-compatible model file, TTAx8, gate/orchestrator code, legend); document global SSIM; disclose earlier non-blind LLM assessor and GT-based gate; 3D -> 2.5D reframing; response letter + tracked-change manuscript.
