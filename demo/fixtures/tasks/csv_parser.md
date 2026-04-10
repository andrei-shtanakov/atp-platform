# Task: CSV Parser

Write a module with three functions:

1. `read_csv(path: str) -> list[dict[str, str]]`
   - Reads a CSV file and returns a list of dicts
   - The first row is the header
   - Raises FileNotFoundError if the file does not exist

2. `filter_rows(data: list[dict], column: str, value: str) -> list[dict]`
   - Filters rows where column == value
   - Raises KeyError if the column does not exist

3. `write_csv(data: list[dict], path: str) -> None`
   - Writes a list of dicts to a CSV file
   - Headers are derived from the keys of the first dict
   - If data is empty, creates an empty file

Use only the standard library (the csv module).
