# Diffusion-Based Probabilistic Gravity Inversion

Research materials for **The Role of Geological Prior Information in Diffusion-Based Probabilistic Gravity Inversion**.

The study compares three geological-information states under a common measurement-guided diffusion inversion workflow: an unconditional prior for unknown geology, a family-conditioned prior for known geology, and an equal-weight conditional scenario ensemble for uncertain geology.

## Contents

- `analysis/notebooks/conditional_ddpm_oracle_equal_mixture.ipynb`: conditional DDPM training, SimPEG baseline, DPS inversion, known-family (oracle) inference, and equal-weight scenario pooling.
- `analysis/notebooks/conditional_results_publication_analysis.ipynb`: manuscript-level metric, statistical, uncertainty, and figure analysis.
- `analysis/artifacts/04_conditional_results.zip`: archived fixed 100 ID + 100 within-family OOD per-case outputs used for the reported results.
- `analysis/data/`: final CSV summaries and paired statistical results.
- `analysis/scripts/`: validation and manuscript-figure regeneration scripts.
- `figures/`: final manuscript figures.
- `environment/` and `requirements.txt`: recorded environment/package information.
- `paper/`: current full manuscript PDF.

## Reproducing the reported analysis

Run `analysis/notebooks/conditional_results_publication_analysis.ipynb` from the repository root. It reads `analysis/artifacts/04_conditional_results.zip` and regenerates manuscript-level metrics and statistical summaries.

The final result figures can be regenerated from the frozen outputs with:

```bash
python analysis/scripts/regenerate_manuscript_figures.py
```

## Environment

The recorded diffusion execution environment used Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, and a Tesla T4 GPU. See `environment/` for the recorded metadata.

## Scope note

The archived outputs are sufficient to reproduce the reported manuscript-level analysis and figures. The conditional experiment notebook expects the generated training/validation/test arrays and the pretrained unconditional checkpoint from the upstream data-generation/unconditional-training pipeline.
