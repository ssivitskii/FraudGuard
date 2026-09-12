# Contributing

Bug reports, documentation fixes and focused improvements are welcome.

## Local setup

Fork [ssivitskii/FraudGuard](https://github.com/ssivitskii/FraudGuard), clone your fork and create a branch. Use Python 3.11 and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Follow [README.md](README.md) for data and runtime configuration. Never commit credentials or private datasets.

## Pull requests

Describe the problem, the resulting behavior and how you checked the change. Keep changes focused, add relevant regression tests for behavior changes and update documentation when commands or interfaces change.

For metric claims, include dataset provenance, preprocessing, split, random seed, model parameters and the command that produced the results. Distinguish measured results from example output.

Report bugs through [Issues](https://github.com/ssivitskii/FraudGuard/issues), including a minimal reproduction, Python version and relevant logs with secrets removed.
