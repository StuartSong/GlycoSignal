# Release Process

## Prerequisites

```bash
pip install build twine
```

## Steps

### 1. Bump the version

Edit `src/glycosignal/__init__.py` and `pyproject.toml` to update the version number:

```
__version__ = "X.Y.Z"
```

```toml
version = "X.Y.Z"
```

### 2. Run tests

```bash
pip install -e ".[dev]"
pytest
```

### 3. Build distribution artifacts

```bash
rm -rf dist/
python -m build
```

This produces both a wheel (`.whl`) and source distribution (`.tar.gz`) in `dist/`.

### 4. Verify the build

```bash
twine check dist/*
```

### 5. Upload to TestPyPI (optional, recommended for first release)

```bash
twine upload --repository testpypi dist/*
```

Install from TestPyPI to verify:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ GlycoSignal
python -c "import glycosignal; print(glycosignal.__version__)"
```

### 6. Upload to PyPI

```bash
twine upload dist/*
```

### 7. Tag the release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 8. Create a GitHub Release (optional)

Go to the repository's Releases page and create a new release from the tag with a changelog summary.
