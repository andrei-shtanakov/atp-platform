"""Tests validating the generated CSV parser."""

import os
import tempfile

import pytest
from csv_parser import filter_rows, read_csv, write_csv

SAMPLE_CSV = "name,age,city\nAlice,30,Moscow\nBob,25,Berlin\nCharlie,30,Moscow\n"


@pytest.fixture
def csv_file(tmp_path: object) -> str:
    """Create a temporary CSV file with sample data."""
    path = os.path.join(tempfile.mkdtemp(), "test.csv")
    with open(path, "w") as f:
        f.write(SAMPLE_CSV)
    return path


class TestReadCsv:
    """Tests for read_csv."""

    def test_reads_file(self, csv_file: str) -> None:
        data = read_csv(csv_file)
        assert len(data) == 3

    def test_returns_list_of_dicts(self, csv_file: str) -> None:
        data = read_csv(csv_file)
        assert isinstance(data, list)
        assert all(isinstance(row, dict) for row in data)

    def test_headers_are_keys(self, csv_file: str) -> None:
        data = read_csv(csv_file)
        assert set(data[0].keys()) == {"name", "age", "city"}

    def test_values_correct(self, csv_file: str) -> None:
        data = read_csv(csv_file)
        assert data[0]["name"] == "Alice"
        assert data[1]["city"] == "Berlin"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_csv("/nonexistent/path.csv")


class TestFilterRows:
    """Tests for filter_rows."""

    def test_filter_by_city(self, csv_file: str) -> None:
        data = read_csv(csv_file)
        result = filter_rows(data, "city", "Moscow")
        assert len(result) == 2
        assert all(row["city"] == "Moscow" for row in result)

    def test_filter_no_match(self, csv_file: str) -> None:
        data = read_csv(csv_file)
        result = filter_rows(data, "city", "Tokyo")
        assert len(result) == 0

    def test_filter_invalid_column(self, csv_file: str) -> None:
        data = read_csv(csv_file)
        with pytest.raises(KeyError):
            filter_rows(data, "nonexistent", "value")


class TestWriteCsv:
    """Tests for write_csv."""

    def test_write_and_read_back(self) -> None:
        data = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"},
        ]
        path = os.path.join(tempfile.mkdtemp(), "output.csv")
        write_csv(data, path)
        result = read_csv(path)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"

    def test_write_empty_data(self) -> None:
        path = os.path.join(tempfile.mkdtemp(), "empty.csv")
        write_csv([], path)
        assert os.path.exists(path)
