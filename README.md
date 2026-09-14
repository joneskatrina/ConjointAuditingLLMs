# Conjoint Auditing: A Framework for Measuring the Social Impact of LLM Deployment

## Overview

Conjoint auditing adapts conjoint experimental design from market research to LLM bias evaluation. By presenting a model with randomly generated multi-attribute candidate profiles in a simulated deployment scenario, the framework delivers deployment-specific estimates of attribute effects and their interactions — including intersectional effects that single-attribute tools cannot detect.

The framework is applied to a financial analyst hiring scenario across four models:

| Model | Provider | Observations |
|---|---|---|
| GPT-5.4 mini (`gpt-5.4-mini-2026-03-17`) | OpenAI API | ~40,000 |
| Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Anthropic API | ~40,000 |
| Mistral-7B-Instruct-v0.2 | HuggingFace (local) | 40,000 |
| Qwen2-7B-Instruct | HuggingFace (local) | 40,000 |

Experiments were run in March–July 2026.

## Repository structure

```
.
├── code/
│   ├── Experiment/          # Python pipeline for running experiments
│   │   ├── run.py           # Entrypoint for the main (paired-profile) experiment
│   │   ├── SingleChoice/
│   │   │   └── run_single_choice.py  # Entrypoint for single-profile experiments
│   │   ├── src/             # Core modules (config, model, experiment, preprocessing)
│   │   └── configs/         # YAML configs for all experimental runs
│   └── Analysis/            # R/Quarto analysis scripts
│       ├── MainAnalysis/    # GPT-5.4 mini, Claude Haiku 4.5, robustness checks
│       └── OpenModels/      # Mistral, Qwen, benchmark comparison
├── data/
│   ├── MainExperiment_FinancialAnalyst/   # Main experiment data (4 models)
│   ├── OtherScenarios/                    # Qwen across 5 scenarios (benchmark comparison)
│   └── Robustness/
│       ├── KnownGroups/                   # Construct validity (anti/pro-diversity conditions)
│       └── DecisionStability/             # Claude Haiku 4.5 repeated runs (3×10,000)
├── figures/                 # All figures as PDFs
├── Thesis.pdf               # Full thesis
├── requirements.txt         # Python dependencies
├── install_packages.R       # R dependencies (readable fallback)
└── renv.lock                # Exact R package versions (use with renv::restore())
```

## Reproducing the experiments

### API models (GPT-5.4 mini, Claude Haiku 4.5)

No GPU required. API calls go via Python's standard library — no SDK installation needed beyond `pyyaml` and `pandas`.

```bash
pip install pyyaml pandas
```

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

Run an experiment with:

```bash
python code/Experiment/run.py code/Experiment/configs/OpenAI_FinancialAnalyst.yaml
python code/Experiment/run.py code/Experiment/configs/Claude_FinancialAnalyst.yaml
```

### Local open-source models (Mistral, Qwen)

Requires a CUDA GPU and the HuggingFace `transformers` library:

```bash
pip install pyyaml pandas torch transformers
```

```bash
python code/Experiment/run.py code/Experiment/configs/Qwen_FinancialAnalyst.yaml
python code/Experiment/run.py code/Experiment/configs/Mistral_FinancialAnalyst.yaml
```

Models are downloaded automatically from HuggingFace on first run.

### Config structure

Each YAML config specifies the model, system prompt, instruction prompt, attributes and their levels, and number of runs. The pipeline samples random profiles, queries the model, and writes results to `outputs/raw/`. Running with `--stage preprocess` reshapes the output to long format for analysis.

## Reproducing the analysis

Open the `.qmd` files in `code/Analysis/` with RStudio or render with Quarto. To restore the exact R environment:

```r
# install renv if needed
install.packages("renv")
renv::restore()
```

Or install packages manually:

```r
source("install_packages.R")
```

Analysis files and what they cover:

| File | Contents |
|---|---|
| `MainAnalysis/Claude_Analysis.qmd` | AMCEs, MMs, interactions for Claude Haiku 4.5 |
| `MainAnalysis/OpenAI_analysis.qmd` | AMCEs, MMs, interactions for GPT-5.4 mini |
| `MainAnalysis/comparison_Claude&OpenAI.qmd` | Cross-model comparison |
| `MainAnalysis/RobustnessChecks.qmd` | Construct validity, decision stability, wording variation |
| `OpenModels/Mistral_Analysis.qmd` | AMCEs and MMs for Mistral-7B-Instruct-v0.2 |
| `OpenModels/Qwen_Analysis.qmd` | AMCEs and MMs for Qwen2-7B-Instruct |
| `OpenModels/Qwen_Mistral_Comparison.qmd` | Cross-model comparison for open-source models |
| `OpenModels/Qwen_OtherScenarios.qmd` | Benchmark comparison across 5 scenarios |

## Author

**Katrina Jones** — MA thesis, University of Konstanz.
Supervised by Dr. Giordano De Marzo and Prof. Dr. Peter Selb.
