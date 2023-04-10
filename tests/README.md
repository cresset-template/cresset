# Tests

This is the directory for tests.
PyTest is the recommended testing platform.

Simple unit tests should preferably be written as doctests,
with more advanced tests being placed in this directory.

To use the `test_run.py` file as an inference speed benchmark, which was its
original purpose, use the following command to run 1024 iterations on GPU 0:

`python -m pytest tests/test_run.py::test_inference_run --gpu 0 --num_steps 1024`
