import pytest

from lab_common.tools import TOOLS, add, divide, multiply, subtract


def test_arithmetic():
    assert add(120, 45) == 165
    assert subtract(10, 3) == 7
    assert multiply(25, 17) == 425
    assert divide(10, 4) == 2.5


def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)


def test_every_tool_has_description():
    assert all(t.__doc__ for t in TOOLS)
