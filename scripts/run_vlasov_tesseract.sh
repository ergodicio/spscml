# Run the Vlasov solver tesseract using the tesseract-runtime API (not in Docker)

PYTHONPATH=src TESSERACT_API_PATH=tesseracts/sheaths/vlasov/tesseract_api.py uv run tesseract-runtime apply @tesseracts/sheaths/example_inputs/apply.json
