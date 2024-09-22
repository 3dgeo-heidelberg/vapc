# content of test_sysexit.py
import pytest


class A:
    def f():
        raise SystemExit(1)

class B:
    def g():
        pass
        raise SystemExit(1)


def test_VASP():
    with pytest.raises(SystemExit):
        A.f()
    with pytest.raises(SystemExit):
        B.g()