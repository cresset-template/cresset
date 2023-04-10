import pytest


def pytest_addoption(parser):
    parser.addoption('--num_steps', type=int, action='store', default=64)
