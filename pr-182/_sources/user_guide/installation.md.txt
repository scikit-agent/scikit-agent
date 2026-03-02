# Installation

## Requirements

scikit-agent requires:

- Python 3.9 or higher
- NumPy
- SciPy
- Matplotlib
- PyTorch
- Cairo

### macOS

To install system dependencies:

```bash
brew install cairo libffi
```

And link Cairo by setting the environment variable:

```
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

## Install from PyPI

```bash
pip install scikit-agent
```

## Install from Source

To install the latest development version:

```bash
git clone https://github.com/scikit-agent/scikit-agent.git
cd scikit-agent
pip install -e .
```

## Development Installation

For development, install with additional dependencies:

```bash
git clone https://github.com/scikit-agent/scikit-agent.git
cd scikit-agent
pip install -e ".[dev,docs]"
```

## Verify Installation

To verify your installation:

```python
import skagent

print(skagent.__version__)
```
