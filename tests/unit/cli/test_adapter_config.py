"""Tests for adapter config parsing."""

from atp.cli.main import _parse_adapter_config, _split_config_item


class TestSplitConfigItem:
    def test_no_comma(self) -> None:
        assert _split_config_item("key=value") == ["key=value"]

    def test_two_pairs(self) -> None:
        assert _split_config_item("a=1,b=2") == ["a=1", "b=2"]

    def test_comma_in_value(self) -> None:
        # URL contains no '=' after the comma → stays together
        assert _split_config_item("url=http://x,y") == ["url=http://x,y"]

    def test_mixed(self) -> None:
        result = _split_config_item("a=1,b=http://x")
        assert result == ["a=1", "b=http://x"]

    def test_url_with_comma_and_second_key(self) -> None:
        result = _split_config_item(
            "endpoint=http://localhost:8001,allow_internal=true"
        )
        assert result == [
            "endpoint=http://localhost:8001",
            "allow_internal=true",
        ]


class TestParseAdapterConfig:
    def test_simple_key_value(self) -> None:
        result = _parse_adapter_config(("key=value",))
        assert result == {"key": "value"}

    def test_multiple_flags(self) -> None:
        result = _parse_adapter_config(("a=1", "b=2"))
        assert result == {"a": 1, "b": 2}  # parsed as JSON ints

    def test_comma_separated(self) -> None:
        result = _parse_adapter_config(
            ("endpoint=http://localhost:8001,allow_internal=true",)
        )
        assert result == {
            "endpoint": "http://localhost:8001",
            "allow_internal": True,  # parsed as JSON bool
        }

    def test_json_value(self) -> None:
        result = _parse_adapter_config(("count=42",))
        assert result == {"count": 42}

    def test_json_list(self) -> None:
        result = _parse_adapter_config(('tools=["a","b"]',))
        assert result == {"tools": ["a", "b"]}

    def test_empty(self) -> None:
        result = _parse_adapter_config(())
        assert result == {}

    def test_url_preserved(self) -> None:
        result = _parse_adapter_config(("endpoint=http://localhost:8001",))
        assert result == {"endpoint": "http://localhost:8001"}
