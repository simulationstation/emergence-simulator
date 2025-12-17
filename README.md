# emergence-simulator

A toy model for exploring hypothetical vacuum bubble configurations across three dimensions: complexity ceilings, formation rarity, and persistence.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Verify installation
python -m emergence_simulator
# Output: emergence-simulator ok

# Run comprehensive sweep
python -m emergence_simulator --sweep-all --fast --outdir outputs

# Generate master report
python -m emergence_simulator --report-master --outdir outputs
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `--bubble-demo` | Run bubble parameter sweep demo |
| `--bubble-dynamics` | Run single bubble dynamics simulation |
| `--sweep-all` | Comprehensive sweep over R, dE, tau, and eta |
| `--report-master` | Generate consolidated report from artifacts |
| `--fast` | Use reduced grid size for faster execution |
| `--outdir DIR` | Output directory (default: `outputs_emergence`) |
| `--eta VALUE` | Background feed coupling constant |
| `--E_bg_J VALUE` | Background energy scale in Joules |

## Core Concepts

### Complexity Ceilings

**Bekenstein Bound** — Maximum information storage:
```
bits = (2π × R × E) / (ℏ × c × ln2)
```

**Lloyd Limit** — Maximum computation rate:
```
ops/s = (2 × E) / (π × ℏ)
```

### Rarity Models

| Model | Formula | Parameters |
|-------|---------|------------|
| Thermal | `logP = -dE / (kB × T)` | T = 2.7 K |
| Instanton A | `logP = -(α × R × dE) / (ℏ × c)` | α = 1.0 |
| Instanton B | `logP = -(α × dE² × R) / (c × E_scale × ℏ)` | α = 1.0, E_scale = 1e-9 J |

### Energy Dynamics

With background feed term:
```
dE/dt = -λE + η × E_bg

Solution: E(t) = E₀ exp(-λt) + (η × E_bg / λ)(1 - exp(-λt))
```

- `λ` (leak_rate): Energy dissipation rate
- `η` (eta): Background feed coupling
- `E_bg`: Background energy scale

### Persistence Classification

Based on window-integrated activity `C_win(T) = ∫[T,2T] f(t) dt`:

| Class | Slope Criterion | Interpretation |
|-------|-----------------|----------------|
| Persistent | slope ≥ -1 | Activity sustained at late times |
| LongTailTerminal | -2 < slope < -1 | Power-law decay |
| Terminal | slope ≤ -2 | Fast exponential decay |

## Output Artifacts

### Sweep (`--sweep-all`)
- `sweep/sweep_results.json` — Full results with all metrics
- `sweep/sweep_report.md` — Analysis report with tables
- `sweep/persistence_heatmap.png` — Classification across (R, dE)
- `sweep/F_inst_a_heatmap.png` — Figure of merit heatmap
- `sweep/ops_vs_rarity_persistence.png` — Scatter by class
- `sweep/pareto_frontier.png` — Optimal trade-offs
- `sweep/fraction_terminal_vs_eta.png` — Classification vs eta

### Dynamics (`--bubble-dynamics`)
- `dynamics/dynamics_results.json` — Simulation data
- `dynamics/dynamics_report.md` — Classification report
- `dynamics/activity_over_time.png` — f(t) and R(t) plots
- `dynamics/window_capacity.png` — C_win(T) with slope

### Master Report (`--report-master`)
- `MASTER_REPORT.md` — Consolidated report with all sections

## Example

```bash
# Full analysis pipeline
python -m emergence_simulator --sweep-all --outdir results
python -m emergence_simulator --bubble-dynamics --outdir results
python -m emergence_simulator --report-master --outdir results

# View results
cat results/MASTER_REPORT.md
```

## Testing

```bash
pytest -q
```

## Limitations

- Toy model for exploratory purposes only
- Equations are illustrative, not derived from first principles
- No uncertainty quantification
- Grid resolution may miss important parameter regimes

## License

MIT
