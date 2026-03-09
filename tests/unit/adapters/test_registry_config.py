"""Tests for config-driven adapter registry."""

from atp.adapters.registry import AdapterRegistry


class TestConfigDrivenRegistry:
    def test_load_from_config(self) -> None:
        reg = AdapterRegistry()
        config = {
            "adapters": {
                "my-http": {
                    "class": "atp.adapters.http.HTTPAdapter",
                    "config_class": "atp.adapters.http.HTTPAdapterConfig",
                },
            },
        }
        count = reg.load_from_config(config)
        assert count == 1
        assert reg.is_registered("my-http")

    def test_load_empty_config(self) -> None:
        reg = AdapterRegistry()
        count = reg.load_from_config({})
        assert count == 0

    def test_skip_invalid_class_path(self) -> None:
        reg = AdapterRegistry()
        config = {
            "adapters": {
                "bad": {"class": "no_dots"},
            },
        }
        count = reg.load_from_config(config)
        assert count == 0
        assert not reg.is_registered("bad")

    def test_default_config_class(self) -> None:
        reg = AdapterRegistry()
        config = {
            "adapters": {
                "custom": {
                    "class": "atp.adapters.http.HTTPAdapter",
                },
            },
        }
        count = reg.load_from_config(config)
        assert count == 1
        assert reg.is_registered("custom")

    def test_load_from_yaml_missing_file(self) -> None:
        reg = AdapterRegistry()
        count = reg.load_from_yaml("/nonexistent/adapters.yaml")
        assert count == 0

    def test_load_from_yaml_real_file(self, tmp_path) -> None:
        config_file = tmp_path / "adapters.yaml"
        config_file.write_text(
            "adapters:\n"
            "  test-adapter:\n"
            '    class: "atp.adapters.http.HTTPAdapter"\n'
            '    config_class: "atp.adapters.http.HTTPAdapterConfig"\n'
        )
        reg = AdapterRegistry()
        count = reg.load_from_yaml(config_file)
        assert count == 1
        assert reg.is_registered("test-adapter")

    def test_multiple_adapters(self) -> None:
        reg = AdapterRegistry()
        config = {
            "adapters": {
                "a": {"class": "atp.adapters.http.HTTPAdapter"},
                "b": {"class": "atp.adapters.cli.CLIAdapter"},
            },
        }
        count = reg.load_from_config(config)
        assert count == 2
