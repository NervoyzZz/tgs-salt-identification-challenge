# tgs-salt-identification-challenge
Competition from Kaggle to segment salt on images. You can read about [here](https://www.kaggle.com/c/tgs-salt-identification-challenge/).

## Repository structure
Repository contains following top-level directories:
* `data` &mdash; contains data for challenge and trained models.
* `production` &mdash; contains production ready code.
* `research` &mdash; contains code that was written during problem solution.

### Data directory
The `data` directory may contain next subdirectories:
* `csv` &mdash; contains `csv` files from Kaggle.
* `models` &mdash; contains models trained during research.

### Production directory
The `production` directory will contain production-ready code.

### Research directory
The `research` directory contains all code developed during problem solution.

Directory may include next subdirectories:
* `analysis` &mdash; scripts and notebooks with analysis of data, parameters, e.t.c.
* `evaluate` &mdash; scripts for evaluation of models and algorithms.
* `tools` &mdash; scripts for data processing.
* `train` &mdash; scripts related to models.

## Coding style and conventions
For this project we follow the next agreements:
1. Use Git Flow branching model.
2. Close issues with commit messages.
3. Use Python 3.* as interpreter.
4. Use single quotes (`'`) for single line string literals in Python code.
5. Use Google style docstrings in Python code.
6. Follow PEP8 and PEP257 recommendations for code and docstring formatting.
7. Use flake8 command line tool to check Python code for issues.
8. Python applications should be made executable. They should have correct 
shebang and executable bit set.
9. Executable Python scripts should have function `main` and their
`if __name__ == '__main__':` block should contain only call to this
function.
10. Use `argparse` module to parse script command line arguments. Do not use
underscores (`_`) in arguments names (use `-` as it usually found in most
*nix apps).


