"""Suite definition routes.

This module provides CRUD operations for suite definitions,
including creating, updating, and exporting suites to YAML.
"""

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from atp.dashboard.models import SuiteDefinition
from atp.dashboard.schemas import (
    AgentConfigCreate,
    AssertionCreate,
    ConstraintsCreate,
    ScoringWeightsCreate,
    SuiteCreateRequest,
    SuiteDefinitionList,
    SuiteDefinitionResponse,
    SuiteDefinitionSummary,
    TaskCreate,
    TestCreateRequest,
    TestDefaultsCreate,
    TestResponse,
    YAMLExportResponse,
)
from atp.dashboard.v2.dependencies import CurrentUser, DBSession, RequiredUser

router = APIRouter(prefix="/suite-definitions", tags=["suite-definitions"])


def _build_suite_definition_response(
    suite_def: SuiteDefinition,
) -> SuiteDefinitionResponse:
    """Build a SuiteDefinitionResponse from a SuiteDefinition model.

    Args:
        suite_def: The database model.

    Returns:
        The response schema.
    """
    # Convert JSON fields back to Pydantic models
    defaults = TestDefaultsCreate(**suite_def.defaults_json)
    agents = [AgentConfigCreate(**a) for a in suite_def.agents_json]
    tests = [
        TestResponse(
            id=t["id"],
            name=t["name"],
            description=t.get("description"),
            tags=t.get("tags", []),
            task=TaskCreate(**t["task"]),
            constraints=ConstraintsCreate(**t.get("constraints", {})),
            assertions=[AssertionCreate(**a) for a in t.get("assertions", [])],
            scoring=ScoringWeightsCreate(**t["scoring"]) if t.get("scoring") else None,
        )
        for t in suite_def.tests_json
    ]

    return SuiteDefinitionResponse(
        id=suite_def.id,
        name=suite_def.name,
        version=suite_def.version,
        description=suite_def.description,
        defaults=defaults,
        agents=agents,
        tests=tests,
        created_at=suite_def.created_at,
        updated_at=suite_def.updated_at,
    )


@router.post(
    "",
    response_model=SuiteDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_suite_definition(
    session: DBSession,
    suite_data: SuiteCreateRequest,
    user: RequiredUser,
) -> SuiteDefinitionResponse:
    """Create a new test suite definition.

    Requires authentication. Creates a new suite definition that can be
    managed through the dashboard and exported to YAML.

    Args:
        session: Database session.
        suite_data: Suite definition data.
        user: Authenticated user.

    Returns:
        The created suite definition.

    Raises:
        HTTPException: If a suite with the same name already exists.
    """
    # Check for existing suite with same name
    stmt = select(SuiteDefinition).where(SuiteDefinition.name == suite_data.name)
    result = await session.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Suite '{suite_data.name}' already exists",
        )

    # Convert Pydantic models to dicts for JSON storage
    defaults_dict = suite_data.defaults.model_dump()
    agents_list = [a.model_dump() for a in suite_data.agents]
    tests_list = [t.model_dump() for t in suite_data.tests]

    # Create suite definition
    suite_def = SuiteDefinition(
        name=suite_data.name,
        version=suite_data.version,
        description=suite_data.description,
        defaults_json=defaults_dict,
        agents_json=agents_list,
        tests_json=tests_list,
        created_by_id=user.id,
    )
    session.add(suite_def)
    await session.commit()
    await session.refresh(suite_def)

    return _build_suite_definition_response(suite_def)


@router.get(
    "",
    response_model=SuiteDefinitionList,
)
async def list_suite_definitions(
    session: DBSession,
    user: CurrentUser,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> SuiteDefinitionList:
    """List all suite definitions.

    Args:
        session: Database session.
        user: Current user (optional auth).
        limit: Maximum items to return.
        offset: Offset for pagination.

    Returns:
        Paginated list of suite definitions.
    """
    # Get total count
    count_stmt = select(func.count(SuiteDefinition.id))
    total = (await session.execute(count_stmt)).scalar() or 0

    # Get paginated results
    stmt = (
        select(SuiteDefinition)
        .order_by(SuiteDefinition.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    suites = result.scalars().all()

    items = [
        SuiteDefinitionSummary(
            id=s.id,
            name=s.name,
            version=s.version,
            description=s.description,
            test_count=s.test_count,
            agent_count=s.agent_count,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in suites
    ]

    return SuiteDefinitionList(
        total=total,
        items=items,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{suite_id}",
    response_model=SuiteDefinitionResponse,
)
async def get_suite_definition(
    session: DBSession,
    suite_id: int,
    user: CurrentUser,
) -> SuiteDefinitionResponse:
    """Get a suite definition by ID.

    Args:
        session: Database session.
        suite_id: Suite definition ID.
        user: Current user (optional auth).

    Returns:
        The suite definition.

    Raises:
        HTTPException: If suite not found.
    """
    suite_def = await session.get(SuiteDefinition, suite_id)
    if suite_def is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite definition {suite_id} not found",
        )

    return _build_suite_definition_response(suite_def)


@router.post(
    "/{suite_id}/tests",
    response_model=SuiteDefinitionResponse,
)
async def add_test_to_suite(
    session: DBSession,
    suite_id: int,
    test_data: TestCreateRequest,
    user: RequiredUser,
) -> SuiteDefinitionResponse:
    """Add a test to an existing suite definition.

    Requires authentication. Adds a new test to the suite's tests list.

    Args:
        session: Database session.
        suite_id: Suite definition ID.
        test_data: Test definition data.
        user: Authenticated user.

    Returns:
        The updated suite definition.

    Raises:
        HTTPException: If suite not found or test ID already exists.
    """
    suite_def = await session.get(SuiteDefinition, suite_id)
    if suite_def is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite definition {suite_id} not found",
        )

    # Check if test ID already exists
    existing_ids = {t.get("id") for t in suite_def.tests_json}
    if test_data.id in existing_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Test ID '{test_data.id}' already exists in suite",
        )

    # Add test to suite
    test_dict = test_data.model_dump()
    suite_def.tests_json = [*suite_def.tests_json, test_dict]

    await session.commit()
    await session.refresh(suite_def)

    return _build_suite_definition_response(suite_def)


@router.delete(
    "/{suite_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_suite_definition(
    session: DBSession,
    suite_id: int,
    user: RequiredUser,
) -> None:
    """Delete a suite definition.

    Requires authentication.

    Args:
        session: Database session.
        suite_id: Suite definition ID.
        user: Authenticated user.

    Raises:
        HTTPException: If suite not found.
    """
    suite_def = await session.get(SuiteDefinition, suite_id)
    if suite_def is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite definition {suite_id} not found",
        )

    await session.delete(suite_def)
    await session.commit()


@router.get(
    "/{suite_id}/yaml",
    response_model=YAMLExportResponse,
)
async def export_suite_yaml(
    session: DBSession,
    suite_id: int,
    user: CurrentUser,
) -> YAMLExportResponse:
    """Export a suite definition as YAML.

    Args:
        session: Database session.
        suite_id: Suite definition ID.
        user: Current user (optional auth).

    Returns:
        YAML content and metadata.

    Raises:
        HTTPException: If suite not found or has no tests.
    """
    suite_def = await session.get(SuiteDefinition, suite_id)
    if suite_def is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite definition {suite_id} not found",
        )

    if not suite_def.tests_json:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot export suite with no tests",
        )

    # Convert to TestSuiteData for YAML export
    from atp.generator.core import TestSuiteData
    from atp.generator.writer import YAMLWriter
    from atp.loader.models import (
        AgentConfig,
        Assertion,
        Constraints,
        ScoringWeights,
        TaskDefinition,
        TestDefaults,
        TestDefinition,
    )

    # Build TestDefaults
    defaults_data = suite_def.defaults_json
    defaults = TestDefaults(
        runs_per_test=defaults_data.get("runs_per_test", 1),
        timeout_seconds=defaults_data.get("timeout_seconds", 300),
        scoring=ScoringWeights(**defaults_data.get("scoring", {})),
        constraints=Constraints(**defaults_data.get("constraints", {}))
        if defaults_data.get("constraints")
        else None,
    )

    # Build agents
    agents = [
        AgentConfig(
            name=a["name"],
            type=a.get("type"),
            config=a.get("config", {}),
        )
        for a in suite_def.agents_json
    ]

    # Build tests
    tests = []
    for t in suite_def.tests_json:
        task_data = t["task"]
        tests.append(
            TestDefinition(
                id=t["id"],
                name=t["name"],
                description=t.get("description"),
                tags=t.get("tags", []),
                task=TaskDefinition(
                    description=task_data["description"],
                    input_data=task_data.get("input_data"),
                    expected_artifacts=task_data.get("expected_artifacts"),
                ),
                constraints=Constraints(**t.get("constraints", {})),
                assertions=[
                    Assertion(type=a["type"], config=a.get("config", {}))
                    for a in t.get("assertions", [])
                ],
                scoring=ScoringWeights(**t["scoring"]) if t.get("scoring") else None,
            )
        )

    # Create suite data
    suite_data = TestSuiteData(
        name=suite_def.name,
        version=suite_def.version,
        description=suite_def.description,
        defaults=defaults,
        agents=agents,
        tests=tests,
    )

    # Generate YAML
    writer = YAMLWriter()
    yaml_content = writer.to_yaml(suite_data)

    return YAMLExportResponse(
        yaml_content=yaml_content,
        suite_name=suite_def.name,
        test_count=len(suite_def.tests_json),
    )
