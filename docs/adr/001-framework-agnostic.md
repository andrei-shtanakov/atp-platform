# ADR-001: Framework Agnostic Design

**Status**: Accepted
**Date**: 2025-01-21
**Decision Makers**: Architecture Team

## Context

AI agents are developed using various frameworks: LangGraph, CrewAI, AutoGen, LangChain, and many others. New frameworks appear regularly, while existing ones become deprecated (for example, AutoGen is discontinuing active development).

We need a testing platform that is not tied to a specific framework.

## Decision

We adopt a **framework-agnostic** approach where the agent is treated as a black box with a defined interaction protocol (ATP Protocol).

### Key Principles

1. **Agent = Black Box**
   - The platform doesn't care how the agent is implemented internally
   - Only the contract matters: input (task) → output (artifacts + metrics)

2. **Protocol as Contract**
   - JSON-based protocol independent of programming language
   - Schemas for validation
   - Versioning for backward compatibility

3. **Adapters for Convenience**
   - Optional adapters for popular frameworks
   - Adapters are syntactic sugar, not mandatory

## Consequences

### Positive

- **Flexibility**: Teams can use any framework
- **Longevity**: Platform won't become obsolete with the framework
- **Comparability**: Can compare agents across different frameworks
- **Easy Integration**: Minimal requirements for the agent
- **Testability**: Protocol can be tested independently

### Negative

- **Overhead**: Need to implement the protocol for each agent
- **Limited Features**: No access to internal framework details
- **Maintenance**: Need to support adapters for popular frameworks

### Risks

- Protocol may not cover all use cases → mitigation: versioning, extensibility
- Adapters may lag behind frameworks → mitigation: community contributions

## Alternatives Considered

### 1. Tight Integration with One Framework
Deep integration with a single framework (e.g., LangGraph).

**Pros**: More capabilities, less overhead
**Cons**: Vendor lock-in, limited audience

**Rejected**: Too risky in a rapidly changing landscape.

### 2. Plugin-per-Framework Architecture
Each framework has its own plugin with full integration.

**Pros**: Maximum capabilities for each framework
**Cons**: Huge maintenance costs, logic duplication

**Rejected**: Doesn't scale.

### 3. Language-Specific Approach
Only Python agents with a defined interface.

**Pros**: Easier to implement
**Cons**: Excludes non-Python agents

**Rejected**: Limits adoption.

## Implementation Notes

1. Protocol is defined in `docs/04-protocol.md`
2. JSON Schema in `schemas/`
3. Adapters in `atp/adapters/`
4. Integration guide in `docs/06-integration.md`

## References

- ATP Protocol Spec: `docs/04-protocol.md`
- Integration Guide: `docs/06-integration.md`
