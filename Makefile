
.PHONY: install lint test train clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	python -m flake8 src/ tests/ --max-line-length=100

test:
	python -m pytest tests/ -v

train:
	python src/train.py

clean:
	python -c "import shutil; from pathlib import Path; root = Path('.'); [shutil.rmtree(path, ignore_errors=True) for path in root.rglob('__pycache__')]; [path.unlink(missing_ok=True) for path in root.rglob('*.pyc')]; shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('temp', ignore_errors=True)"