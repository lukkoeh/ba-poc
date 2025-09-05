.PHONY: install run clean
install:
	pip install -r requirements.txt
run:
	python run.py all --transcripts data/transcripts --meta data/meta
clean:
	rm -rf artifacts/* artifacts/.[!.]* || true
