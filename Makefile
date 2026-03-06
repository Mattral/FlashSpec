.PHONY: lint format test test-gpu test-chaos bench bench-quick docs docs-serve \
        paper docker-build docker-test clean

lint:
	ruff check flashspec benchmarks tests scripts
	ruff format --check flashspec benchmarks tests scripts
	mypy --strict flashspec
	lint-imports

format:
	ruff format flashspec benchmarks tests scripts

test:
	pytest tests/unit tests/integration -x --cov=flashspec --cov-report=term-missing

test-gpu:
	pytest tests/ -m gpu -x --cov=flashspec

test-chaos:
	pytest tests/chaos -x

bench:
	python benchmarks/run_all.py --config benchmarks/configs/

bench-quick:
	python benchmarks/run_all.py --config benchmarks/configs/ --toy

docs:
	mkdocs build

docs-serve:
	mkdocs serve

paper:
	cd paper && make

docker-build:
	docker build -t flashspec:latest .

docker-test:
	docker run flashspec:latest make test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov dist build *.egg-info
	find . -name "*.pyc" -delete
