# Installation

## Requirements

scikit-agent requires Python 3.9 or higher. Its main runtime dependencies are:

- NumPy, SciPy, SymPy, Numba, and xarray
- PyTorch
- Matplotlib
- cairosvg, pydot, PyYAML, and IPython (model visualization)

See `pyproject.toml` for the authoritative dependency list; `pip` installs all
of these automatically.

### System Dependencies

The `cairosvg` and `pydot` packages link against the Cairo and Graphviz system
libraries, which must be installed separately for model visualization.

On macOS:

```bash
brew install cairo libffi graphviz
```

And link Cairo by setting the environment variable:

```
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

On Debian/Ubuntu Linux:

```bash
sudo apt-get install libcairo2 graphviz
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
