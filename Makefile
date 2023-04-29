run_tests:
	poetry run pytest tests/

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run mypy .
	poetry run black . --check
	poetry run isort . --check
	poetry run flake8 .

start:
	poetry run uvicorn bullpen_backend.main:app --reload --port 8001