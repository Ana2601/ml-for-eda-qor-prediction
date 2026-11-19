.PHONY: demo test clean

demo:
	python3 qor_model.py

test:
	python3 test_ml_eda.py

clean:
	rm -rf data results __pycache__