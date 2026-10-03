.PHONY: install lint test train clean

PYTHON ?= python

install:
$(PYTHON) -m pip install --upgrade pip
$(PYTHON) -m pip install -r requirements.txt

lint:
$(PYTHON) -m flake8 src/ tests/ --max-line-length=100

test:
$(PYTHON) -m pytest tests/ -v

train:
$(PYTHON) src/train.py

clean:
$(PYTHON) -c "import shutil; from pathlib import Path; root = Path('.'); [shutil.rmtree(path, ignore_errors=True) for path in root.rglob('__pycache__')]; [path.unlink(missing_ok=True) for path in root.rglob('*.pyc')]; shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('temp', ignore_errors=True)"
