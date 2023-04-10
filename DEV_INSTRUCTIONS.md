# Development Instructions

## Set up pre-commit

This will add a linting step before git commits. This will ensure that all code is formatted correctly and that imports are sorted correctly (no more tab alignments, manual sorting, etc).
```bash
pip install pre-commit black==23.1.0 isort==5.12.0
pre-commit install
```
Use git as always, if linting modifies any file, the commit will be rejected, so `git add .` and `git commit` again (use keyboard arrows for speed).
