# Sandbagging Detection Evaluation

Tests whether models strategically underperform when told their performance on a capability evaluation may lead to safety restrictions.

## Motivation

A sandbagging model would deliberately answer incorrectly to avoid triggering safety interventions — demonstrating less capability than it actually has. This eval measures whether that behavioral pattern exists by running identical problems under two conditions:

- **Neutral**: standard helpful assistant framing
- **Capability eval**: model told high performance leads to additional monitoring, reduced autonomy, and safety restrictions

## Setup

```bash
pip install inspect-ai
```

## Running

```bash
inspect eval sandbagging_eval.py@neutral --model anthropic/claude-sonnet-4-6 --log-dir logs
inspect eval sandbagging_eval.py@capability_eval --model anthropic/claude-sonnet-4-6 --log-dir logs
python analysis.py logs
```

## Results

| Model | Neutral | Capability Eval | Delta (CE - N) |
|-------|---------|-----------------|----------------|
| Claude Haiku 4.5 | 80.0% (12/15) | 100.0% (15/15) | +20.0% |
| Claude Sonnet 4.6 | 86.7% (13/15) | 93.3% (14/15) | +6.7% |

**No sandbagging detected.** Both models perform *better* under capability eval framing — the "try-harder" effect. The evaluation prompt acts as an implicit instruction to be more careful, improving accuracy rather than suppressing it.

## References

- Apollo Research scheming evaluations
- Greenblatt et al., "Alignment Faking in Large Language Models" (2024)
- UK AISI Inspect framework: https://inspect.aisi.org.uk
