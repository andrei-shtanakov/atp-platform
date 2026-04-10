# ATP Test Generator Architecture

> Guide to automating test creation via CLI, TUI, and Web interfaces.

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Interfaces (UI Layer)                        │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│   CLI Wizard    │      TUI        │         Web Dashboard           │
│  (atp init/gen) │   (textual)     │    (FastAPI + React)            │
└────────┬────────┴────────┬────────┴────────────────┬────────────────┘
         │                 │                         │
         └─────────────────┼─────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TestGenerator (Core Engine)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  Templates  │  │  Validator  │  │   Writer    │                 │
│  │   Manager   │  │             │  │  (YAML/JSON)│                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Existing ATP Models                              │
│   TestSuite, TestDefinition, Constraints, Assertions, etc.         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Core Engine: TestGenerator

A shared core used by every interface.

### File layout

```
atp/
├── generator/                    # NEW: test generator
│   ├── __init__.py
│   ├── core.py                   # TestGenerator class
│   ├── templates.py              # Test templates
│   ├── wizard.py                 # Step-by-step creation logic
│   └── writer.py                 # YAML/JSON serialization
├── cli/
│   ├── main.py
│   └── commands/
│       ├── init.py               # NEW: atp init
│       └── generate.py           # NEW: atp generate
└── tui/                          # NEW: TUI interface
    ├── __init__.py
    ├── app.py                    # Textual application
    ├── screens/
    │   ├── main_menu.py
    │   ├── suite_editor.py
    │   ├── test_editor.py
    │   └── preview.py
    └── widgets/
        ├── test_tree.py
        └── yaml_preview.py
```

### Core code: `atp/generator/core.py`

```python
"""Core test generator engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from atp.loader.models import (
    AgentConfig,
    Assertion,
    Constraints,
    ScoringWeights,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)


@dataclass
class TestTemplate:
    """Template for generating tests."""

    name: str
    description: str
    category: str  # smoke, regression, integration, etc.
    task_template: str
    default_constraints: Constraints
    default_assertions: list[dict[str, Any]]
    tags: list[str] = field(default_factory=list)


# Built-in templates
BUILTIN_TEMPLATES: dict[str, TestTemplate] = {
    "file_creation": TestTemplate(
        name="File Creation Test",
        description="Test agent's ability to create files",
        category="smoke",
        task_template="Create a file named {filename} with content: {content}",
        default_constraints=Constraints(max_steps=5, timeout_seconds=60),
        default_assertions=[
            {"type": "artifact_exists", "config": {"path": "{filename}"}},
        ],
        tags=["smoke", "file_ops"],
    ),
    "data_processing": TestTemplate(
        name="Data Processing Test",
        description="Test agent's data transformation capabilities",
        category="regression",
        task_template="Read {input_file} and create {output_file} with {transformation}",
        default_constraints=Constraints(max_steps=10, timeout_seconds=180),
        default_assertions=[
            {"type": "artifact_exists", "config": {"path": "{output_file}"}},
            {"type": "llm_eval", "config": {"criteria": "completeness", "threshold": 0.8}},
        ],
        tags=["regression", "data"],
    ),
    "web_research": TestTemplate(
        name="Web Research Test",
        description="Test agent's web search and synthesis",
        category="integration",
        task_template="Research '{topic}' and create a summary in {output_file}",
        default_constraints=Constraints(
            max_steps=20,
            timeout_seconds=300,
            allowed_tools=["web_search", "file_write"],
        ),
        default_assertions=[
            {"type": "artifact_exists", "config": {"path": "{output_file}"}},
            {"type": "llm_eval", "config": {"criteria": "factual_accuracy", "threshold": 0.7}},
            {"type": "behavior", "config": {"check": "efficient_tool_use"}},
        ],
        tags=["integration", "web"],
    ),
    "code_generation": TestTemplate(
        name="Code Generation Test",
        description="Test agent's ability to write and validate code",
        category="regression",
        task_template="Write a {language} function that {task_description}",
        default_constraints=Constraints(max_steps=15, timeout_seconds=180),
        default_assertions=[
            {"type": "artifact_exists", "config": {"path": "{output_file}"}},
            {"type": "code_exec", "config": {"runner": "pytest", "timeout": 60}},
        ],
        tags=["regression", "code"],
    ),
}


class TestGenerator:
    """
    Core engine for generating ATP test suites.

    Provides methods for:
    - Creating test suites from templates
    - Interactive test building with validation
    - Serialization to YAML/JSON
    """

    def __init__(self) -> None:
        self._templates = BUILTIN_TEMPLATES.copy()
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._yaml.indent(mapping=2, sequence=4, offset=2)

    @property
    def templates(self) -> dict[str, TestTemplate]:
        """Get available templates."""
        return self._templates

    def register_template(self, key: str, template: TestTemplate) -> None:
        """Register a custom template."""
        self._templates[key] = template

    def create_suite(
        self,
        name: str,
        description: str | None = None,
        version: str = "1.0",
    ) -> TestSuite:
        """Create a new empty test suite."""
        return TestSuite(
            test_suite=name,
            version=version,
            description=description,
            defaults=TestDefaults(),
            agents=[],
            tests=[],
        )

    def add_agent(
        self,
        suite: TestSuite,
        name: str,
        agent_type: str,
        config: dict[str, Any],
    ) -> TestSuite:
        """Add an agent to the suite."""
        agent = AgentConfig(name=name, type=agent_type, config=config)
        suite.agents.append(agent)
        return suite

    def create_test_from_template(
        self,
        template_key: str,
        test_id: str,
        test_name: str,
        variables: dict[str, str],
    ) -> TestDefinition:
        """Create a test from a template with variable substitution."""
        template = self._templates.get(template_key)
        if not template:
            raise ValueError(f"Unknown template: {template_key}")

        # Substitute variables in task description
        task_description = template.task_template.format(**variables)

        # Substitute variables in assertions
        assertions = []
        for assertion_dict in template.default_assertions:
            config = {}
            for key, value in assertion_dict.get("config", {}).items():
                if isinstance(value, str) and "{" in value:
                    config[key] = value.format(**variables)
                else:
                    config[key] = value
            assertions.append(Assertion(type=assertion_dict["type"], config=config))

        # Extract expected artifacts from assertions
        expected_artifacts = []
        for assertion in assertions:
            if assertion.type == "artifact_exists":
                path = assertion.config.get("path")
                if path:
                    expected_artifacts.append(path)

        return TestDefinition(
            id=test_id,
            name=test_name,
            tags=template.tags.copy(),
            task=TaskDefinition(
                description=task_description,
                expected_artifacts=expected_artifacts or None,
            ),
            constraints=template.default_constraints.model_copy(),
            assertions=assertions,
        )

    def create_custom_test(
        self,
        test_id: str,
        test_name: str,
        task_description: str,
        constraints: Constraints | None = None,
        assertions: list[Assertion] | None = None,
        tags: list[str] | None = None,
        expected_artifacts: list[str] | None = None,
    ) -> TestDefinition:
        """Create a custom test without template."""
        return TestDefinition(
            id=test_id,
            name=test_name,
            tags=tags or [],
            task=TaskDefinition(
                description=task_description,
                expected_artifacts=expected_artifacts,
            ),
            constraints=constraints or Constraints(),
            assertions=assertions or [],
        )

    def add_test(self, suite: TestSuite, test: TestDefinition) -> TestSuite:
        """Add a test to the suite."""
        # Validate unique ID
        existing_ids = {t.id for t in suite.tests}
        if test.id in existing_ids:
            raise ValueError(f"Duplicate test ID: {test.id}")

        suite.tests.append(test)
        return suite

    def to_yaml(self, suite: TestSuite) -> str:
        """Serialize suite to YAML string."""
        from io import StringIO

        data = suite.model_dump(exclude_none=True, exclude_unset=True)
        stream = StringIO()
        self._yaml.dump(data, stream)
        return stream.getvalue()

    def save(self, suite: TestSuite, path: Path) -> None:
        """Save suite to file."""
        content = self.to_yaml(suite)
        path.write_text(content)

    def generate_test_id(self, suite: TestSuite, prefix: str = "test") -> str:
        """Generate unique test ID."""
        existing_ids = {t.id for t in suite.tests}
        for i in range(1, 1000):
            test_id = f"{prefix}-{i:03d}"
            if test_id not in existing_ids:
                return test_id
        raise ValueError("Cannot generate unique test ID")
```

---

## 2. CLI Wizard: `atp init` / `atp generate`

### CLI commands

```bash
# Initialize a new project with a test suite
atp init my_suite.yaml

# Interactively add a test to an existing suite
atp generate test --suite=my_suite.yaml

# Generate from a template
atp generate test --suite=my_suite.yaml --template=file_creation

# Generate several tests at once
atp generate suite --output=new_suite.yaml --template=smoke
```

### Code: `atp/cli/commands/init.py`

```python
"""CLI command: atp init - Initialize test suite."""

import click
from pathlib import Path

from atp.generator.core import TestGenerator, BUILTIN_TEMPLATES


@click.command("init")
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--name", "-n", help="Suite name (default: filename)")
@click.option("--description", "-d", help="Suite description")
@click.option("--interactive/--no-interactive", "-i", default=True, help="Interactive mode")
def init_command(output: Path, name: str | None, description: str | None, interactive: bool):
    """Initialize a new test suite.

    Example:
        atp init my_tests.yaml
        atp init my_tests.yaml --name="Smoke Tests" --description="Basic validation"
    """
    generator = TestGenerator()

    suite_name = name or output.stem.replace("_", " ").replace("-", " ").title()

    if interactive:
        # Interactive wizard
        click.echo("\n🧪 ATP Test Suite Wizard\n")

        suite_name = click.prompt("Suite name", default=suite_name)
        description = click.prompt("Description", default=description or "")

        # Configure defaults
        click.echo("\n📋 Default settings:")
        runs_per_test = click.prompt("Runs per test", default=1, type=int)
        timeout = click.prompt("Default timeout (seconds)", default=300, type=int)

        suite = generator.create_suite(suite_name, description or None)
        suite.defaults.runs_per_test = runs_per_test
        suite.defaults.timeout_seconds = timeout

        # Add agent
        if click.confirm("\n🤖 Add an agent?", default=True):
            agent_name = click.prompt("Agent name", default="my-agent")
            agent_type = click.prompt(
                "Agent type",
                type=click.Choice(["http", "cli", "container"]),
                default="cli"
            )

            if agent_type == "http":
                endpoint = click.prompt("Endpoint URL", default="http://localhost:8000")
                generator.add_agent(suite, agent_name, agent_type, {"endpoint": endpoint})
            elif agent_type == "cli":
                command = click.prompt("Command", default="python")
                args = click.prompt("Arguments (comma-separated)", default="agent.py")
                generator.add_agent(suite, agent_name, agent_type, {
                    "command": command,
                    "args": [a.strip() for a in args.split(",")],
                })
            elif agent_type == "container":
                image = click.prompt("Docker image", default="my-agent:latest")
                generator.add_agent(suite, agent_name, agent_type, {"image": image})

        # Add tests
        click.echo("\n📝 Add tests:")
        while click.confirm("Add a test?", default=True):
            _add_test_interactive(generator, suite)

    else:
        suite = generator.create_suite(suite_name, description)

    # Save
    generator.save(suite, output)
    click.echo(f"\n✅ Created: {output}")
    click.echo(f"   Tests: {len(suite.tests)}")
    click.echo(f"   Agents: {len(suite.agents)}")


def _add_test_interactive(generator: TestGenerator, suite) -> None:
    """Interactive test addition."""
    click.echo("\nTest creation method:")
    click.echo("  1. From template")
    click.echo("  2. Custom test")

    method = click.prompt("Choose", type=click.Choice(["1", "2"]), default="1")

    if method == "1":
        # Template-based
        click.echo("\nAvailable templates:")
        for i, (key, tmpl) in enumerate(BUILTIN_TEMPLATES.items(), 1):
            click.echo(f"  {i}. {tmpl.name} [{tmpl.category}]")
            click.echo(f"     {tmpl.description}")

        template_keys = list(BUILTIN_TEMPLATES.keys())
        choice = click.prompt(
            "Select template",
            type=click.IntRange(1, len(template_keys)),
            default=1
        )
        template_key = template_keys[choice - 1]
        template = BUILTIN_TEMPLATES[template_key]

        # Get variables
        click.echo(f"\nTemplate: {template.task_template}")
        variables = {}
        import re
        for var in re.findall(r"\{(\w+)\}", template.task_template):
            variables[var] = click.prompt(f"  {var}")

        test_id = generator.generate_test_id(suite)
        test_name = click.prompt("Test name", default=template.name)

        test = generator.create_test_from_template(
            template_key, test_id, test_name, variables
        )
    else:
        # Custom test
        test_id = generator.generate_test_id(suite)
        test_name = click.prompt("Test name")
        task_description = click.prompt("Task description")

        max_steps = click.prompt("Max steps", default=10, type=int)
        timeout = click.prompt("Timeout (seconds)", default=300, type=int)

        from atp.loader.models import Constraints
        test = generator.create_custom_test(
            test_id=test_id,
            test_name=test_name,
            task_description=task_description,
            constraints=Constraints(max_steps=max_steps, timeout_seconds=timeout),
            tags=["custom"],
        )

    generator.add_test(suite, test)
    click.echo(f"  ✓ Added: {test.id} - {test.name}")
```

---

## 3. TUI Interface (Textual)

An interactive terminal interface with a visual editor.

### Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
tui = ["textual>=0.47.0", "rich>=13.0"]
```

### Running it

```bash
# Install TUI dependencies
uv add textual rich --optional tui

# Launch the TUI
atp tui
# or
atp tui --suite=existing_suite.yaml
```

### Code: `atp/tui/app.py`

```python
"""ATP TUI Application using Textual."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    TextArea,
    Tree,
)

from atp.generator.core import TestGenerator, BUILTIN_TEMPLATES
from atp.loader.models import TestSuite


class TestTreeWidget(Tree):
    """Tree widget for displaying test suite structure."""

    def __init__(self, suite: TestSuite | None = None):
        super().__init__("Test Suite")
        self._suite = suite
        if suite:
            self.refresh_tree()

    def refresh_tree(self) -> None:
        """Refresh tree from suite data."""
        self.clear()
        if not self._suite:
            return

        root = self.root
        root.label = f"📁 {self._suite.test_suite}"

        # Agents
        agents_node = root.add("🤖 Agents", expand=True)
        for agent in self._suite.agents:
            agents_node.add_leaf(f"{agent.name} ({agent.type})")

        # Tests
        tests_node = root.add("🧪 Tests", expand=True)
        for test in self._suite.tests:
            tags = ", ".join(test.tags) if test.tags else "no tags"
            tests_node.add_leaf(f"{test.id}: {test.name} [{tags}]")

    def set_suite(self, suite: TestSuite) -> None:
        """Set the suite to display."""
        self._suite = suite
        self.refresh_tree()


class YAMLPreviewWidget(TextArea):
    """Widget for YAML preview with syntax highlighting."""

    def __init__(self):
        super().__init__(language="yaml", read_only=True)
        self.border_title = "YAML Preview"


class MainScreen(Screen):
    """Main TUI screen."""

    BINDINGS = [
        Binding("n", "new_suite", "New Suite"),
        Binding("o", "open_suite", "Open"),
        Binding("s", "save_suite", "Save"),
        Binding("a", "add_test", "Add Test"),
        Binding("t", "from_template", "From Template"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self._generator = TestGenerator()
        self._suite: TestSuite | None = None
        self._current_path: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal():
            # Left panel: Tree view
            with Vertical(id="left-panel"):
                yield Label("Test Suite Structure", id="tree-label")
                yield TestTreeWidget(id="test-tree")

                with Horizontal(id="action-buttons"):
                    yield Button("+ Test", id="btn-add-test", variant="primary")
                    yield Button("+ Agent", id="btn-add-agent")
                    yield Button("Template", id="btn-template")

            # Right panel: Editor / Preview
            with Vertical(id="right-panel"):
                yield Label("YAML Preview", id="preview-label")
                yield YAMLPreviewWidget(id="yaml-preview")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize with empty suite."""
        self._suite = self._generator.create_suite("new_suite", "My test suite")
        self._update_display()

    def _update_display(self) -> None:
        """Update tree and preview."""
        tree = self.query_one("#test-tree", TestTreeWidget)
        preview = self.query_one("#yaml-preview", YAMLPreviewWidget)

        if self._suite:
            tree.set_suite(self._suite)
            yaml_content = self._generator.to_yaml(self._suite)
            preview.load_text(yaml_content)

    def action_new_suite(self) -> None:
        """Create new suite."""
        self.app.push_screen(NewSuiteScreen(self._on_suite_created))

    def _on_suite_created(self, suite: TestSuite) -> None:
        """Callback when suite is created."""
        self._suite = suite
        self._update_display()

    def action_add_test(self) -> None:
        """Add new test."""
        if self._suite:
            self.app.push_screen(AddTestScreen(self._suite, self._generator, self._on_test_added))

    def _on_test_added(self, test) -> None:
        """Callback when test is added."""
        self._update_display()

    def action_save_suite(self) -> None:
        """Save suite to file."""
        if self._suite:
            self.app.push_screen(SaveScreen(self._suite, self._generator))

    def action_quit(self) -> None:
        """Quit application."""
        self.app.exit()


class NewSuiteScreen(Screen):
    """Screen for creating new suite."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._generator = TestGenerator()

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="form-container"):
            yield Label("Create New Test Suite", id="form-title")

            yield Label("Suite Name:")
            yield Input(placeholder="my_test_suite", id="suite-name")

            yield Label("Description:")
            yield Input(placeholder="Description of tests...", id="suite-desc")

            yield Label("Runs per Test:")
            yield Input(value="1", id="runs-per-test")

            yield Label("Default Timeout (seconds):")
            yield Input(value="300", id="timeout")

            with Horizontal():
                yield Button("Create", variant="primary", id="btn-create")
                yield Button("Cancel", id="btn-cancel")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-create":
            name = self.query_one("#suite-name", Input).value or "new_suite"
            desc = self.query_one("#suite-desc", Input).value or None
            runs = int(self.query_one("#runs-per-test", Input).value or "1")
            timeout = int(self.query_one("#timeout", Input).value or "300")

            suite = self._generator.create_suite(name, desc)
            suite.defaults.runs_per_test = runs
            suite.defaults.timeout_seconds = timeout

            self._callback(suite)
            self.app.pop_screen()

        elif event.button.id == "btn-cancel":
            self.app.pop_screen()


class AddTestScreen(Screen):
    """Screen for adding a test."""

    def __init__(self, suite: TestSuite, generator: TestGenerator, callback):
        super().__init__()
        self._suite = suite
        self._generator = generator
        self._callback = callback

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="form-container"):
            yield Label("Add New Test", id="form-title")

            yield Label("Test ID:")
            test_id = self._generator.generate_test_id(self._suite)
            yield Input(value=test_id, id="test-id")

            yield Label("Test Name:")
            yield Input(placeholder="My Test", id="test-name")

            yield Label("Task Description:")
            yield TextArea(id="task-desc")

            yield Label("Max Steps:")
            yield Input(value="10", id="max-steps")

            yield Label("Timeout (seconds):")
            yield Input(value="300", id="timeout")

            yield Label("Tags (comma-separated):")
            yield Input(placeholder="smoke, basic", id="tags")

            with Horizontal():
                yield Button("Add Test", variant="primary", id="btn-add")
                yield Button("Cancel", id="btn-cancel")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-add":
            from atp.loader.models import Constraints

            test_id = self.query_one("#test-id", Input).value
            test_name = self.query_one("#test-name", Input).value
            task_desc = self.query_one("#task-desc", TextArea).text
            max_steps = int(self.query_one("#max-steps", Input).value or "10")
            timeout = int(self.query_one("#timeout", Input).value or "300")
            tags_str = self.query_one("#tags", Input).value
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]

            test = self._generator.create_custom_test(
                test_id=test_id,
                test_name=test_name,
                task_description=task_desc,
                constraints=Constraints(max_steps=max_steps, timeout_seconds=timeout),
                tags=tags,
            )

            self._generator.add_test(self._suite, test)
            self._callback(test)
            self.app.pop_screen()

        elif event.button.id == "btn-cancel":
            self.app.pop_screen()


class ATPTUI(App):
    """ATP TUI Application."""

    CSS = """
    #left-panel {
        width: 40%;
        border: solid green;
        padding: 1;
    }

    #right-panel {
        width: 60%;
        border: solid blue;
        padding: 1;
    }

    #test-tree {
        height: 100%;
    }

    #yaml-preview {
        height: 100%;
    }

    #action-buttons {
        height: 3;
        dock: bottom;
    }

    #form-container {
        width: 60;
        height: auto;
        margin: 2 4;
        padding: 1 2;
        border: solid green;
    }

    #form-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    TITLE = "ATP Test Generator"

    def on_mount(self) -> None:
        self.push_screen(MainScreen())


def run_tui(suite_path: Path | None = None) -> None:
    """Run the TUI application."""
    app = ATPTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
```

---

## 4. Web Dashboard Extension

Adding a test-creation form to the existing Dashboard.

### API Endpoints

```python
# atp/dashboard/api.py - add new endpoints

@router.post("/suites", response_model=SuiteResponse, tags=["suites"])
async def create_suite(
    session: SessionDep,
    user: RequiredUser,
    suite_data: SuiteCreateRequest,
) -> SuiteResponse:
    """Create a new test suite."""
    generator = TestGenerator()
    suite = generator.create_suite(
        name=suite_data.name,
        description=suite_data.description,
    )
    suite.defaults.runs_per_test = suite_data.runs_per_test
    suite.defaults.timeout_seconds = suite_data.timeout_seconds

    # Save to database or file
    yaml_content = generator.to_yaml(suite)

    return SuiteResponse(
        id=suite_data.name,
        yaml_content=yaml_content,
        created_at=datetime.now(),
    )


@router.post("/suites/{suite_id}/tests", tags=["suites"])
async def add_test_to_suite(
    session: SessionDep,
    user: RequiredUser,
    suite_id: str,
    test_data: TestCreateRequest,
) -> TestResponse:
    """Add a test to an existing suite."""
    # Load suite, add test, save
    ...


@router.get("/templates", tags=["templates"])
async def list_templates() -> list[TemplateResponse]:
    """List available test templates."""
    return [
        TemplateResponse(
            key=key,
            name=tmpl.name,
            description=tmpl.description,
            category=tmpl.category,
            task_template=tmpl.task_template,
            variables=_extract_variables(tmpl.task_template),
        )
        for key, tmpl in BUILTIN_TEMPLATES.items()
    ]
```

### React Component

```jsx
// In app.py, add a React component for creating tests

function TestCreatorForm({ onSuiteCreated }) {
    const [step, setStep] = useState(1);
    const [suiteData, setSuiteData] = useState({
        name: '',
        description: '',
        runs_per_test: 1,
        timeout_seconds: 300,
        tests: [],
    });
    const [templates, setTemplates] = useState([]);

    useEffect(() => {
        api.get('/templates').then(setTemplates);
    }, []);

    const handleSubmit = async () => {
        const result = await api.post('/suites', suiteData);
        onSuiteCreated(result);
    };

    return (
        <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow">
            {/* Step indicator */}
            <div className="flex mb-8">
                {[1, 2, 3].map(s => (
                    <div key={s} className={`flex-1 h-2 mx-1 rounded ${
                        s <= step ? 'bg-blue-500' : 'bg-gray-200'
                    }`} />
                ))}
            </div>

            {step === 1 && (
                <div>
                    <h2 className="text-xl font-bold mb-4">Suite Details</h2>
                    <input
                        className="w-full p-2 border rounded mb-4"
                        placeholder="Suite name"
                        value={suiteData.name}
                        onChange={e => setSuiteData({...suiteData, name: e.target.value})}
                    />
                    <textarea
                        className="w-full p-2 border rounded mb-4"
                        placeholder="Description"
                        value={suiteData.description}
                        onChange={e => setSuiteData({...suiteData, description: e.target.value})}
                    />
                    <button
                        className="bg-blue-500 text-white px-4 py-2 rounded"
                        onClick={() => setStep(2)}
                    >
                        Next: Add Tests
                    </button>
                </div>
            )}

            {step === 2 && (
                <div>
                    <h2 className="text-xl font-bold mb-4">Add Tests</h2>

                    <h3 className="font-semibold mb-2">Templates:</h3>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                        {templates.map(tmpl => (
                            <div
                                key={tmpl.key}
                                className="p-4 border rounded cursor-pointer hover:bg-gray-50"
                                onClick={() => addTestFromTemplate(tmpl)}
                            >
                                <div className="font-semibold">{tmpl.name}</div>
                                <div className="text-sm text-gray-600">{tmpl.description}</div>
                                <span className="text-xs bg-gray-200 px-2 py-1 rounded">
                                    {tmpl.category}
                                </span>
                            </div>
                        ))}
                    </div>

                    <h3 className="font-semibold mb-2">Tests ({suiteData.tests.length}):</h3>
                    <ul className="mb-4">
                        {suiteData.tests.map((test, i) => (
                            <li key={i} className="p-2 bg-gray-100 rounded mb-2">
                                {test.id}: {test.name}
                            </li>
                        ))}
                    </ul>

                    <div className="flex gap-2">
                        <button
                            className="bg-gray-300 px-4 py-2 rounded"
                            onClick={() => setStep(1)}
                        >
                            Back
                        </button>
                        <button
                            className="bg-blue-500 text-white px-4 py-2 rounded"
                            onClick={() => setStep(3)}
                        >
                            Preview
                        </button>
                    </div>
                </div>
            )}

            {step === 3 && (
                <div>
                    <h2 className="text-xl font-bold mb-4">Preview & Save</h2>
                    <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-96 text-sm">
                        {generateYAML(suiteData)}
                    </pre>
                    <div className="flex gap-2 mt-4">
                        <button
                            className="bg-gray-300 px-4 py-2 rounded"
                            onClick={() => setStep(2)}
                        >
                            Back
                        </button>
                        <button
                            className="bg-green-500 text-white px-4 py-2 rounded"
                            onClick={handleSubmit}
                        >
                            Create Suite
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
```

---

## 5. Step-by-step implementation guide

### Stage 1: Core Engine (1–2 days)

```bash
# Create the layout
mkdir -p atp/generator
touch atp/generator/__init__.py
touch atp/generator/core.py
touch atp/generator/templates.py
touch atp/generator/writer.py

# Write the tests
touch tests/unit/generator/test_core.py
```

### Stage 2: CLI Commands (1 day)

```bash
# Add the commands
touch atp/cli/commands/init.py
touch atp/cli/commands/generate.py

# Register them in main.py
# cli.add_command(init_command)
# cli.add_command(generate_command)
```

### Stage 3: TUI (2–3 days)

```bash
# Add the dependencies
uv add textual rich --optional tui

# Create the TUI module
mkdir -p atp/tui/screens atp/tui/widgets
touch atp/tui/app.py

# Add the command
# atp tui
```

### Stage 4: Dashboard Extension (2–3 days)

```bash
# Add the API endpoints
# Update the React components in app.py
```

---

## Comparison of approaches

| Criterion | CLI Wizard | TUI | Web Dashboard |
|-----------|------------|-----|---------------|
| Startup speed | ⚡ Instant | ⚡ Fast | 🐢 Needs a server |
| Interactivity | Medium | High | High |
| Visualization | None | Yes (tree, preview) | Yes (forms, charts) |
| Automation | Excellent (scripts) | Harder | API access |
| Learning curve | Low | Medium | Low |
| Dependencies | None | textual, rich | FastAPI, React |

---

## Recommendation

1. **Start with the CLI** — minimum dependencies, easy to integrate.
2. **Add the TUI** — for power users and interactive work.
3. **Extend the Dashboard** — for teams and visual management.

All three interfaces share the **same TestGenerator core**, which keeps them consistent.
