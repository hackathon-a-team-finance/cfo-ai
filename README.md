# CFO AI

TODO

## Local development

### Poetry

Create the virtual environment and install dependencies with:

```shell
poetry install
```

To run formatting and linting:

```shell
make format; make lint
```

Run commands inside the virtual environment with:

```shell
poetry run <your_command>
```

for example, to run a script:

```shell
poetry run python3 cfo_ai/scripts/test_script.py
```

To run streamlit app locally:

```shell
poetry run streamlit run cfo_ai/streamlit/local_app.py
```

To run the tests:

```
make run_tests
```

To use a shell with virtual environment:

```shell
poetry shell
```

Start a development server locally:

```shell
poetry run uvicorn app.main:app --reload --host localhost --port 8001
```

or

```shell
make start
```
