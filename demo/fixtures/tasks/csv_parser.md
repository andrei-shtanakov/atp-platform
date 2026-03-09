# Задание: CSV Parser

Напиши модуль с тремя функциями:

1. `read_csv(path: str) -> list[dict[str, str]]`
   - Читает CSV-файл, возвращает список словарей
   - Первая строка — заголовки
   - Поднимает FileNotFoundError если файл не найден

2. `filter_rows(data: list[dict], column: str, value: str) -> list[dict]`
   - Фильтрует строки где column == value
   - Поднимает KeyError если column не существует

3. `write_csv(data: list[dict], path: str) -> None`
   - Записывает список словарей в CSV-файл
   - Заголовки берёт из ключей первого словаря
   - Если data пуст — создаёт пустой файл

Используй только стандартную библиотеку (модуль csv).
