# Running tests

Before running tests, change working directory to `env\environment`.

To run a particular test, execute

```bash
python -m unittest discover -s tests -p 'test_daycounting_conventions.py'
```

To run all tests, execute

```bash
python -m unittest discover -s tests
```

or use file matching pattern if your test filenames don't start with 'test_'

```bash
python -m unittest discover -s tests -p '*_test.py'
```
