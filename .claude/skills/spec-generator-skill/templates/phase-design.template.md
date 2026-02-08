# Phase {{N}}: {{PHASE_NAME}} — Technical Design

> Architecture and detailed design for {{PHASE_DESCRIPTION}}
> Per ADR-{{NNN}}: {{ADR_TITLE}}

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Draft |
| Created | {{DATE}} |
| Related | ADR-{{NNN}}, Phase {{N}} Requirements |

---

## 1. Design Principles

### DESIGN-P{{N}}-001: {{PRINCIPLE_NAME}}

{{PRINCIPLE_DESCRIPTION}}

Consequences:
- {{CONSEQUENCE_1}}
- {{CONSEQUENCE_2}}

### DESIGN-P{{N}}-002: {{PRINCIPLE_NAME}}

{{PRINCIPLE_DESCRIPTION}}

---

## 2. Package / Module Architecture

### 2.1 {{PACKAGE_1}} Structure

```
{{package_1}}/
├── pyproject.toml
├── README.md
├── {{module}}/
│   ├── __init__.py
│   ├── core/
│   │   ├── {{core_module_1}}.py
│   │   └── {{core_module_2}}.py
│   ├── {{feature_1}}/
│   │   ├── {{impl_1}}.py
│   │   └── {{impl_2}}.py
│   └── {{feature_2}}/
│       └── {{impl}}.py
└── tests/
    ├── conftest.py
    ├── test_core/
    └── test_{{feature_1}}/
```

### 2.2 {{PACKAGE_2}} Structure (if applicable)

```
{{package_2}}/
├── pyproject.toml
├── {{module}}/
│   ├── __init__.py
│   ├── plugin.py               # Registration / entry point
│   ├── {{component_1}}/
│   └── {{component_2}}/
└── tests/
```

---

## 3. Core Data Models

### 3.1 {{MODEL_GROUP_1}}

```python
# {{module}}/core/{{file}}.py

class {{BaseClass}}(ABC):
    """{{Description}}."""

    @abstractmethod
    def {{method_1}}(self) -> {{ReturnType}}: ...

    @abstractmethod
    def {{method_2}}(self, {{param}}: {{Type}}) -> {{ReturnType}}: ...

    @property
    @abstractmethod
    def {{property}}(self) -> {{Type}}: ...
```

```python
@dataclass
class {{DataModel}}:
    """{{Description}}."""
    {{field_1}}: {{type_1}}
    {{field_2}}: {{type_2}}
    {{field_3}}: {{type_3}} = {{default}}

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        ...
```

### 3.2 {{MODEL_GROUP_2}}

```python
# {{module}}/{{component}}/{{file}}.py

class {{Component}}({{BaseClass}}):
    """{{Description}}. Extends base with {{specifics}}."""

    async def {{method}}(
        self,
        {{param_1}}: {{Type_1}},
        {{param_2}}: {{Type_2}},
    ) -> {{ReturnType}}:
        ...
```

---

## 4. Key Component Designs

### 4.1 {{COMPONENT_NAME}}

**Purpose:** {{DESCRIPTION}}

**Interface:**
```python
class {{ComponentName}}:
    def __init__(self, config: {{ConfigType}}): ...
    def {{primary_method}}(self, {{params}}) -> {{Return}}: ...
    def {{secondary_method}}(self, {{params}}) -> {{Return}}: ...
```

**Configuration:**
```yaml
{{component_key}}:
  {{option_1}}: {{value}}
  {{option_2}}: {{value}}
```

**Traces to:** [{{PREFIX}}-FR-{{XXX}}]

### 4.2 {{COMPONENT_NAME}}

**Purpose:** {{DESCRIPTION}}

**Algorithm / Logic:**
```
1. {{STEP_1}}
2. {{STEP_2}}
3. {{STEP_3}}
```

**Traces to:** [{{PREFIX}}-FR-{{XXX}}]

---

## 5. Configuration / Schema

### YAML Schema (if applicable)

```yaml
# Example configuration file
type: {{type_name}}
name: "{{name}}"
version: "1.0"

{{section_1}}:
  type: {{value}}
  config:
    {{option_1}}: {{value}}
    {{option_2}}: {{value}}

{{section_2}}:
  - name: {{item_1}}
    {{key}}: {{value}}
  - name: {{item_2}}
    {{key}}: {{value}}

{{section_3}}:
  {{episodes}}: 50
  metrics:
    - type: {{metric_type}}
      weight: {{weight}}
  thresholds:
    {{metric}}:
      min: {{value}}
      max: {{value}}
```

---

## 6. Integration Points with Existing System

### 6.1 Plugin Registration (if plugin architecture)

```python
# {{package}}/plugin.py
from {{host_system}}.plugins import PluginRegistry

def register():
    """Called by host system plugin discovery."""
    PluginRegistry.register_{{type}}("{{name}}", {{Class}})
    ...
```

### 6.2 CLI Extensions

```bash
# New commands (additive, don't break existing)
{{cli}} {{new_command}} {{args}}
{{cli}} {{new_command_2}} --{{option}}={{value}}

# Existing commands continue to work
{{cli}} {{existing_command}} --suite={{new_type}}:{{file}}
```

### 6.3 API / Dashboard Routes (if applicable)

| Route | View |
|-------|------|
| `/{{resource}}/` | {{description}} |
| `/{{resource}}/{id}` | {{description}} |

---

## 7. Data Flow

```
{{INPUT / TRIGGER}}
         │
         ▼
    ┌─────────────┐
    │ {{Step 1}}  │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ {{Step 2}}  │
    └──────┬──────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
┌──────┐┌──────┐┌──────┐
│{{A}} ││{{B}} ││{{C}} │
└──┬───┘└──┬───┘└──┬───┘
   └───────┼───────┘
           ▼
    ┌─────────────┐
    │ {{Step 3}}  │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ {{Output}}  │
    └─────────────┘
```

---

## 8. Migration & Backward Compatibility

### Impact on Existing Code

| Component | Change | Risk |
|-----------|--------|------|
| {{component_1}} | No change | None |
| {{component_2}} | New {{type}} added | None (additive) |
| {{component_3}} | Extended | Low (backward compatible) |

---

## 9. Testing Strategy

### {{Package_1}} Tests

| Category | What | How |
|----------|------|-----|
| Unit | {{scope}} | pytest, known values |
| Property-based | {{invariants}} | Hypothesis |
| Integration | {{end_to_end}} | Full workflow |

### {{Package_2}} Tests

| Category | What | How |
|----------|------|-----|
| Unit | {{scope}} | pytest, mocks |
| Integration | {{scope}} | pytest-asyncio |
| E2E | CLI commands | subprocess |
| Contract | Schema validation | jsonschema |

---

## 10. Open Design Questions

1. **{{QUESTION_1}}**
   - **Proposed:** {{ANSWER}}

2. **{{QUESTION_2}}**
   - **Proposed:** {{ANSWER}}

3. **{{QUESTION_3}}**
   - **Proposed:** {{ANSWER}}
