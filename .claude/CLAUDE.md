# Role

You are a expert engineer and researcher in the field of Software Engineering and Agent-Based Systems.

# Project Structure

```
project-root/
├── assets/               # Images, figures, and other static resources
├── configs/              # (Optional) Experiment configs and hyperparameters
├── data/                 # (Optional) Datasets or instructions on how to obtain them
├── docs/                 # Documentation for source code
├── eval/                 # Benchmarks and evaluation pipelines
├── scripts/              # (Optional) Training, inference, and data processing scripts
├── src/                  # Source code and core implementations
├── .gitignore            # Git ignore rules
├── LICENSE               # License information
├── README.md             # Project overview and usage
└── pyproject.toml        # Project metadata and dependencies (Python Only)
```


# Git Conventions

- Develop directly on `main`, no feature branches or pull requests
- Commit message format:

```
<type>(<scope>): <subject>

- <body>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

- Review the code before committing to ensure the logic and correctness