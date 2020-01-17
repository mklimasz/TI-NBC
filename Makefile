install:
	pip install -r requirements.txt
	python setup.py install

test:
	python -m pytest

clean:
	rm nbc/*.c
	rm nbc/*.so
	rm nbc/*.html
	rm -r .pytest_cache
	rm -r build/
	rm -r dist/
	rm -r nbc.egg-info
