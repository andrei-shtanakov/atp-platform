"""Test template routes.

This module provides endpoints for discovering and listing
available test templates that can be used to create tests.
"""

from fastapi import APIRouter

from atp.dashboard.schemas import (
    AssertionCreate,
    ConstraintsCreate,
    TemplateListResponse,
    TemplateResponse,
)
from atp.dashboard.v2.dependencies import CurrentUser

router = APIRouter(prefix="/templates", tags=["templates"])


@router.get(
    "",
    response_model=TemplateListResponse,
)
async def list_templates(
    user: CurrentUser,
    category: str | None = None,
) -> TemplateListResponse:
    """List available test templates.

    Returns all registered templates that can be used to create tests.
    Templates provide pre-defined patterns with variable placeholders.

    Args:
        user: Current user (optional auth).
        category: Optional category filter.

    Returns:
        List of templates and available categories.
    """
    from atp.generator.templates import (
        TemplateRegistry,
        get_template_variables,
    )

    registry = TemplateRegistry()
    template_names = registry.list_templates()

    templates: list[TemplateResponse] = []
    categories_set: set[str] = set()

    for name in template_names:
        template = registry.get(name)
        categories_set.add(template.category)

        # Filter by category if specified
        if category and template.category != category:
            continue

        # Get variables used in template
        variables = list(get_template_variables(template))

        # Convert constraints to schema
        constraints = ConstraintsCreate(
            max_steps=template.default_constraints.max_steps,
            max_tokens=template.default_constraints.max_tokens,
            timeout_seconds=template.default_constraints.timeout_seconds,
            allowed_tools=template.default_constraints.allowed_tools,
            budget_usd=template.default_constraints.budget_usd,
        )

        # Convert assertions to schema
        assertions = [
            AssertionCreate(type=a.type, config=a.config)
            for a in template.default_assertions
        ]

        templates.append(
            TemplateResponse(
                name=template.name,
                description=template.description,
                category=template.category,
                task_template=template.task_template,
                default_constraints=constraints,
                default_assertions=assertions,
                tags=template.tags,
                variables=variables,
            )
        )

    return TemplateListResponse(
        templates=templates,
        categories=sorted(categories_set),
        total=len(templates),
    )
