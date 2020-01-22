install:
	pip install -r requirements.txt
	python setup.py install

test:
	python -m pytest

clean:
	rm -f nbc/*.c
	rm -f nbc/*.so
	rm -f nbc/*.html
	rm -fr .pytest_cache
	rm -fr build/
	rm -fr dist/
	rm -fr nbc.egg-info
