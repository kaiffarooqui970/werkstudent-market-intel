.PHONY: install ingest transform train dashboard demo test lint clean all

DB=data/warehouse.duckdb

install:
	pip install -e ".[dev]"

ingest:
	python -m src.ingest.arbeitnow --db $(DB)

transform:
	python -m src.transform.run_sql --db $(DB)

train:
	python -m src.ml.salary_model --db $(DB) --out data/model.pkl

dashboard:
	streamlit run src/dashboard/app.py

demo:
	python -m src.ingest.load_sample --db $(DB)
	$(MAKE) transform
	$(MAKE) train
	$(MAKE) dashboard

test:
	pytest -v

lint:
	ruff check src tests

clean:
	rm -rf data/*.duckdb data/*.pkl __pycache__ .pytest_cache
	find . -name "*.pyc" -delete

all: install ingest transform train
