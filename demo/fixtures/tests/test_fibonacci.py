"""Tests validating the generated fibonacci function."""

import pytest

# Import from the file the agent will generate
from fibonacci import fibonacci


class TestFibonacciBasic:
    """Basic cases."""

    def test_zero(self) -> None:
        assert fibonacci(0) == 0

    def test_one(self) -> None:
        assert fibonacci(1) == 1

    def test_two(self) -> None:
        assert fibonacci(2) == 1

    def test_ten(self) -> None:
        assert fibonacci(10) == 55

    def test_twenty(self) -> None:
        assert fibonacci(20) == 6765


class TestFibonacciEdgeCases:
    """Edge cases."""

    def test_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            fibonacci(-1)

    def test_negative_large_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            fibonacci(-100)

    def test_large_number(self) -> None:
        """fibonacci(50) should work without issues."""
        result = fibonacci(50)
        assert result == 12586269025

    def test_return_type_is_int(self) -> None:
        assert isinstance(fibonacci(10), int)


class TestFibonacciSequence:
    """Sequence verification."""

    def test_sequence_first_ten(self) -> None:
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        actual = [fibonacci(i) for i in range(10)]
        assert actual == expected
