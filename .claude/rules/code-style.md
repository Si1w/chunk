---
description: This file describes the code style guidelines for the project.
paths: 
  - "**/*.py"
  - "**/*.rs"
---

# Code Style Guidelines

## Idioms
- "Keep It Simple, Stupid"
- Use existing code as much as possible
- Readability is the first priority when writing code
- Code style should be consistent across the project
- All code should be well-documented with docstrings in English
- All variables should be meaningful, concise, and short

## Evaluation
- Always provide a `--num_samples` argument to allow small-scale dry runs (e.g., ~50) before full evaluation; omitting it runs all instances
- For custom (non-official) eval implementations, separate each step (e.g., data loading, inference, scoring) into independent stages and verify each before proceeding
- Reproducibility is the first priority for evaluation
