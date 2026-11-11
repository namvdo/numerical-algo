# numerical-algo

Hand-implemented numerical linear algebra algorithms in Python (QR factorization, Schur decomposition, SVD, Sylvester equation solvers, etc.).

## Prerequisites

- Python `>= 3.11`

## Setup

Clone the repository and create a local virtual environment:

```bash
git clone https://github.com/namvdo/numerical-algo.git
cd numerical-algo
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode along with development tools (`pytest`, `jupyter`, `ipykernel`):

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

Alternatively, install dependencies directly from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest
```

## Project Layout

```text
numerical-algo/
├── pyproject.toml
├── requirements.txt
├── src/                   # Place implementations here (e.g. back_substitution.py, qr.py, schur.py, svd.py, sylvester.py)
└── tests/                 # Unit tests & reference checks against scipy.linalg / mpmath / sympy
```

## Key Packages

- `numpy` — Array/matrix representation, slicing, and IEEE-754 floating-point primitives
- `numba` — `@njit` compilation for fast scalar/loop-based implementations (Householder, Givens, QR iteration, Golub–Kahan, Bartels–Stewart)
- `scipy` — Reference LAPACK routines (`scipy.linalg.qr`, `schur`, `svd`, `solve_sylvester`) for validation
- `mpmath` / `sympy` — Arbitrary-precision and symbolic arithmetic for conditioning and stability analysis
- `matplotlib` — Convergence and residual error plots
- `pytest` — Automated testing for orthogonality, backward error, and reconstruction accuracy

