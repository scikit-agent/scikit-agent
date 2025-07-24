# Installation

## Requirements

scikit-agent requires:

- Python (>=3.9)
- NumPy
- SciPy  
- SymPy
- Numba
- PyTorch
- PyYAML
- XArray
- PyDot

## Install from PyPI

```bash
pip install scikit-agent
```

## Install from source

To install the latest development version:

```bash
git clone https://github.com/scikit-agent/scikit-agent.git
cd scikit-agent
pip install -e .
```

## Verify installation

```python
import skagent
print(skagent.__version__)
```