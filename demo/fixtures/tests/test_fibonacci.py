"""Тесты для проверки сгенерированной функции fibonacci."""

import pytest

# Импорт из файла, который сгенерирует агент
from fibonacci import fibonacci


class TestFibonacciBasic:
    """Базовые случаи."""

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
    """Граничные случаи."""

    def test_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            fibonacci(-1)

    def test_negative_large_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            fibonacci(-100)

    def test_large_number(self) -> None:
        """fibonacci(50) должен работать без проблем."""
        result = fibonacci(50)
        assert result == 12586269025

    def test_return_type_is_int(self) -> None:
        assert isinstance(fibonacci(10), int)


class TestFibonacciSequence:
    """Проверка последовательности."""

    def test_sequence_first_ten(self) -> None:
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        actual = [fibonacci(i) for i in range(10)]
        assert actual == expected
