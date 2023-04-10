# Development Instructions

## Set up pre-commit

This will add a linting step before git commits. This will ensure that all code is formatted correctly and that imports are sorted correctly (no more tab alignments, manual sorting, etc).
```bash
pip install pre-commit black==23.1.0 isort==5.12.0
pre-commit install
```
Use git as always, if linting modifies any file, the commit will be rejected, so `git add .` and `git commit` again (use keyboard arrows for speed).

## For Training, Download Data:

After `git clone <repo>` and `cd <repo>`, run:

```
mkdir -p data/cath
mkdir -p data/ts
wget -N data/cath.zip https://github.com/A4Bio/PiFold/releases/download/Training%26Data/cath4.2.zip
unzip -o data/cath/cath4.2.zip -d data/cath
mv data/cath/cath4.2/* data/cath/

wget -N data/ts.zip https://github.com/A4Bio/PiFold/releases/download/Training%26Data/ts.zip
unzip -o data/ts.zip -d data/

# download the pre-trained model
mkdir -p results/PiFold
wget -N results/PiFold/checkpoint.pth https://github.com/A4Bio/PiFold/releases/download/Training%26Data/checkpoint.pth
```
