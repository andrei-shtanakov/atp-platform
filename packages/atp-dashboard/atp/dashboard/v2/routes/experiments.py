"""A/B Testing experiment routes.

This module provides endpoints for managing A/B testing experiments,
including CRUD operations, traffic routing, and result analysis.

Permissions:
    - GET /experiments: ANALYTICS_READ
    - GET /experiments/{id}: ANALYTICS_READ
    - POST /experiments: ANALYTICS_WRITE
    - PUT /experiments/{id}: ANALYTICS_WRITE
    - DELETE /experiments/{id}: ANALYTICS_DELETE
    - POST /experiments/{id}/start: ANALYTICS_WRITE
    - POST /experiments/{id}/pause: ANALYTICS_WRITE
    - POST /experiments/{id}/resume: ANALYTICS_WRITE
    - POST /experiments/{id}/conclude: ANALYTICS_WRITE
    - POST /experiments/{id}/observations: ANALYTICS_WRITE
    - GET /experiments/{id}/report: ANALYTICS_READ
    - GET /experiments/{id}/assign: ANALYTICS_READ
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from atp.analytics.ab_testing import (
    Experiment,
    ExperimentConfig,
    ExperimentObservation,
    ExperimentStatus,
    MetricConfig,
    MetricType,
    RollbackConfig,
    Variant,
    VariantType,
    get_experiment_manager,
)
from atp.dashboard.rbac import Permission, require_permission
from atp.dashboard.schemas import (
    ExperimentCreate,
    ExperimentList,
    ExperimentReportResponse,
    ExperimentResponse,
    ExperimentSummary,
    ExperimentUpdate,
    MetricConfigCreate,
    ObservationCreate,
    ObservationResponse,
    RollbackConfigCreate,
    StatisticalResultResponse,
    TrafficAssignmentResponse,
    VariantMetricsResponse,
    VariantResponse,
)
from atp.dashboard.v2.dependencies import RequiredUser

router = APIRouter(prefix="/experiments", tags=["experiments"])


# ==================== Helper Functions ====================


def _experiment_to_response(experiment: Experiment) -> ExperimentResponse:
    """Convert an Experiment to ExperimentResponse."""
    return ExperimentResponse(
        id=experiment.id or 0,
        name=experiment.config.name,
        description=experiment.config.description,
        suite_name=experiment.config.suite_name,
        test_ids=experiment.config.test_ids,
        status=experiment.status.value,
        control_variant=VariantResponse(
            name=experiment.config.control_variant.name,
            variant_type=experiment.config.control_variant.variant_type.value,
            agent_name=experiment.config.control_variant.agent_name,
            traffic_weight=experiment.config.control_variant.traffic_weight,
            description=experiment.config.control_variant.description,
        ),
        treatment_variant=VariantResponse(
            name=experiment.config.treatment_variant.name,
            variant_type=experiment.config.treatment_variant.variant_type.value,
            agent_name=experiment.config.treatment_variant.agent_name,
            traffic_weight=experiment.config.treatment_variant.traffic_weight,
            description=experiment.config.treatment_variant.description,
        ),
        metrics=[
            MetricConfigCreate(
                metric_type=m.metric_type.value,
                is_primary=m.is_primary,
                minimize=m.minimize,
                min_effect_size=m.min_effect_size,
            )
            for m in experiment.config.metrics
        ],
        rollback=RollbackConfigCreate(
            enabled=experiment.config.rollback.enabled,
            degradation_threshold=experiment.config.rollback.degradation_threshold,
            min_samples_before_rollback=experiment.config.rollback.min_samples_before_rollback,
            consecutive_checks=experiment.config.rollback.consecutive_checks,
        ),
        min_sample_size=experiment.config.min_sample_size,
        max_sample_size=experiment.config.max_sample_size,
        max_duration_days=experiment.config.max_duration_days,
        significance_level=experiment.config.significance_level,
        control_sample_size=experiment.control_sample_size,
        treatment_sample_size=experiment.treatment_sample_size,
        winner=experiment.winner.value if experiment.winner else None,
        rollback_triggered=experiment.rollback_triggered,
        created_at=experiment.created_at,
        started_at=experiment.started_at,
        concluded_at=experiment.concluded_at,
        conclusion_reason=experiment.conclusion_reason,
    )


def _experiment_to_summary(experiment: Experiment) -> ExperimentSummary:
    """Convert an Experiment to ExperimentSummary."""
    return ExperimentSummary(
        id=experiment.id or 0,
        name=experiment.config.name,
        suite_name=experiment.config.suite_name,
        status=experiment.status.value,
        control_variant_name=experiment.config.control_variant.name,
        treatment_variant_name=experiment.config.treatment_variant.name,
        control_sample_size=experiment.control_sample_size,
        treatment_sample_size=experiment.treatment_sample_size,
        winner=experiment.winner.value if experiment.winner else None,
        created_at=experiment.created_at,
        started_at=experiment.started_at,
        concluded_at=experiment.concluded_at,
    )


# ==================== Endpoints ====================


@router.get("", response_model=ExperimentList)
async def list_experiments(
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    status: str | None = Query(None, description="Filter by status"),
    suite_name: str | None = Query(None, description="Filter by suite name"),
) -> ExperimentList:
    """List all A/B experiments.

    Requires ANALYTICS_READ permission.

    Args:
        status: Optional filter by experiment status.
        suite_name: Optional filter by suite name.

    Returns:
        ExperimentList with matching experiments.
    """
    manager = get_experiment_manager()

    status_enum = ExperimentStatus(status) if status else None
    experiments = manager.list_experiments(status=status_enum, suite_name=suite_name)

    return ExperimentList(
        items=[_experiment_to_summary(e) for e in experiments],
        total=len(experiments),
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
) -> ExperimentResponse:
    """Get an experiment by ID.

    Requires ANALYTICS_READ permission.

    Args:
        experiment_id: Experiment ID.

    Returns:
        ExperimentResponse with experiment details.
    """
    manager = get_experiment_manager()
    experiment = manager.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return _experiment_to_response(experiment)


@router.post("", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    request: ExperimentCreate,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
    user: RequiredUser,
) -> ExperimentResponse:
    """Create a new A/B experiment.

    Requires ANALYTICS_WRITE permission.

    Args:
        request: Experiment configuration.
        user: Authenticated user (required).

    Returns:
        Created ExperimentResponse.
    """
    manager = get_experiment_manager()

    # Convert request to config
    config = ExperimentConfig(
        name=request.name,
        description=request.description,
        suite_name=request.suite_name,
        test_ids=request.test_ids,
        control_variant=Variant(
            name=request.control_variant.name,
            variant_type=VariantType.CONTROL,
            agent_name=request.control_variant.agent_name,
            traffic_weight=request.control_variant.traffic_weight,
            description=request.control_variant.description,
        ),
        treatment_variant=Variant(
            name=request.treatment_variant.name,
            variant_type=VariantType.TREATMENT,
            agent_name=request.treatment_variant.agent_name,
            traffic_weight=request.treatment_variant.traffic_weight,
            description=request.treatment_variant.description,
        ),
        metrics=[
            MetricConfig(
                metric_type=MetricType(m.metric_type),
                is_primary=m.is_primary,
                minimize=m.minimize,
                min_effect_size=m.min_effect_size,
            )
            for m in request.metrics
        ],
        rollback=RollbackConfig(
            enabled=request.rollback.enabled,
            degradation_threshold=request.rollback.degradation_threshold,
            min_samples_before_rollback=request.rollback.min_samples_before_rollback,
            consecutive_checks=request.rollback.consecutive_checks,
        ),
        min_sample_size=request.min_sample_size,
        max_sample_size=request.max_sample_size,
        max_duration_days=request.max_duration_days,
        significance_level=request.significance_level,
    )

    experiment = manager.create_experiment(config)
    return _experiment_to_response(experiment)


@router.put("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: int,
    request: ExperimentUpdate,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
    user: RequiredUser,
) -> ExperimentResponse:
    """Update an experiment.

    Requires ANALYTICS_WRITE permission.
    Only draft experiments can be updated.

    Args:
        experiment_id: Experiment ID.
        request: Update data.
        user: Authenticated user (required).

    Returns:
        Updated ExperimentResponse.
    """
    manager = get_experiment_manager()
    experiment = manager.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.status != ExperimentStatus.DRAFT:
        raise HTTPException(
            status_code=400,
            detail="Only draft experiments can be updated",
        )

    # Apply updates
    if request.description is not None:
        experiment.config.description = request.description

    if request.rollback is not None:
        experiment.config.rollback = RollbackConfig(
            enabled=request.rollback.enabled,
            degradation_threshold=request.rollback.degradation_threshold,
            min_samples_before_rollback=request.rollback.min_samples_before_rollback,
            consecutive_checks=request.rollback.consecutive_checks,
        )

    if request.max_sample_size is not None:
        experiment.config.max_sample_size = request.max_sample_size

    if request.max_duration_days is not None:
        experiment.config.max_duration_days = request.max_duration_days

    return _experiment_to_response(experiment)


@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(
    experiment_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_DELETE))],
    user: RequiredUser,
) -> None:
    """Delete an experiment.

    Requires ANALYTICS_DELETE permission.
    Running experiments cannot be deleted.

    Args:
        experiment_id: Experiment ID.
        user: Authenticated user (required).
    """
    manager = get_experiment_manager()
    experiment = manager.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.status == ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running experiment. Pause or conclude it first.",
        )

    # Remove from manager
    if experiment_id in manager._experiments:
        del manager._experiments[experiment_id]


@router.post("/{experiment_id}/start", response_model=ExperimentResponse)
async def start_experiment(
    experiment_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
    user: RequiredUser,
) -> ExperimentResponse:
    """Start an experiment.

    Requires ANALYTICS_WRITE permission.
    Transitions experiment from draft to running.

    Args:
        experiment_id: Experiment ID.
        user: Authenticated user (required).

    Returns:
        Updated ExperimentResponse.
    """
    manager = get_experiment_manager()

    try:
        experiment = manager.start_experiment(experiment_id)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{experiment_id}/pause", response_model=ExperimentResponse)
async def pause_experiment(
    experiment_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
    user: RequiredUser,
) -> ExperimentResponse:
    """Pause a running experiment.

    Requires ANALYTICS_WRITE permission.

    Args:
        experiment_id: Experiment ID.
        user: Authenticated user (required).

    Returns:
        Updated ExperimentResponse.
    """
    manager = get_experiment_manager()

    try:
        experiment = manager.pause_experiment(experiment_id)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{experiment_id}/resume", response_model=ExperimentResponse)
async def resume_experiment(
    experiment_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
    user: RequiredUser,
) -> ExperimentResponse:
    """Resume a paused experiment.

    Requires ANALYTICS_WRITE permission.

    Args:
        experiment_id: Experiment ID.
        user: Authenticated user (required).

    Returns:
        Updated ExperimentResponse.
    """
    manager = get_experiment_manager()

    try:
        experiment = manager.resume_experiment(experiment_id)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{experiment_id}/conclude", response_model=ExperimentResponse)
async def conclude_experiment(
    experiment_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
    user: RequiredUser,
    reason: str = Query(default="Manual conclusion", description="Conclusion reason"),
    force: bool = Query(
        default=False, description="Force conclusion without min samples"
    ),
) -> ExperimentResponse:
    """Conclude an experiment.

    Requires ANALYTICS_WRITE permission.
    Performs final analysis and determines winner.

    Args:
        experiment_id: Experiment ID.
        user: Authenticated user (required).
        reason: Reason for conclusion.
        force: Whether to force conclusion without minimum samples.

    Returns:
        Updated ExperimentResponse.
    """
    manager = get_experiment_manager()
    experiment = manager.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if not force and not experiment.can_conclude:
        min_samples = experiment.config.min_sample_size
        ctrl_samples = experiment.control_sample_size
        treat_samples = experiment.treatment_sample_size
        raise HTTPException(
            status_code=400,
            detail=(
                f"Experiment needs at least {min_samples} samples per variant. "
                f"Current: control={ctrl_samples}, treatment={treat_samples}. "
                f"Use force=true to override."
            ),
        )

    try:
        experiment = manager.conclude_experiment(experiment_id, reason)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{experiment_id}/observations", response_model=ObservationResponse)
async def record_observation(
    experiment_id: int,
    request: ObservationCreate,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_WRITE))],
) -> ObservationResponse:
    """Record an observation for an experiment.

    Requires ANALYTICS_WRITE permission.
    Records the result of a test run assigned to a variant.

    Args:
        experiment_id: Experiment ID.
        request: Observation data.

    Returns:
        Created ObservationResponse.
    """
    manager = get_experiment_manager()
    experiment = manager.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    observation = ExperimentObservation(
        experiment_id=experiment_id,
        variant_name=request.variant_name,
        test_id=request.test_id,
        run_id=request.run_id,
        score=request.score,
        success=request.success,
        duration_seconds=request.duration_seconds,
        cost_usd=request.cost_usd,
        tokens=request.tokens,
        metadata=request.metadata,
    )

    try:
        manager.record_observation(experiment_id, observation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Find the observation in the experiment
    all_obs = experiment.control_observations + experiment.treatment_observations
    for obs in reversed(all_obs):
        if obs.run_id == request.run_id:
            return ObservationResponse(
                id=hash(obs.run_id) % 2147483647,  # Generate pseudo-ID
                experiment_id=experiment_id,
                variant_name=obs.variant_name,
                test_id=obs.test_id,
                run_id=obs.run_id,
                timestamp=obs.timestamp,
                score=obs.score,
                success=obs.success,
                duration_seconds=obs.duration_seconds,
                cost_usd=obs.cost_usd,
                tokens=obs.tokens,
            )

    raise HTTPException(status_code=500, detail="Failed to record observation")


@router.get("/{experiment_id}/report", response_model=ExperimentReportResponse)
async def get_experiment_report(
    experiment_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
) -> ExperimentReportResponse:
    """Get comprehensive experiment report.

    Requires ANALYTICS_READ permission.

    Args:
        experiment_id: Experiment ID.

    Returns:
        ExperimentReportResponse with analysis results.
    """
    manager = get_experiment_manager()

    try:
        report = manager.generate_report(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ExperimentReportResponse(
        experiment=_experiment_to_response(report.experiment),
        statistical_results=[
            StatisticalResultResponse(
                metric_type=r.metric_type.value,
                control_metrics=VariantMetricsResponse(
                    variant_name=r.control_metrics.variant_name,
                    sample_size=r.control_metrics.sample_size,
                    mean=r.control_metrics.mean,
                    std=r.control_metrics.std,
                    min_value=r.control_metrics.min_value,
                    max_value=r.control_metrics.max_value,
                    ci_lower=r.control_metrics.ci_lower,
                    ci_upper=r.control_metrics.ci_upper,
                ),
                treatment_metrics=VariantMetricsResponse(
                    variant_name=r.treatment_metrics.variant_name,
                    sample_size=r.treatment_metrics.sample_size,
                    mean=r.treatment_metrics.mean,
                    std=r.treatment_metrics.std,
                    min_value=r.treatment_metrics.min_value,
                    max_value=r.treatment_metrics.max_value,
                    ci_lower=r.treatment_metrics.ci_lower,
                    ci_upper=r.treatment_metrics.ci_upper,
                ),
                t_statistic=r.t_statistic,
                p_value=r.p_value,
                is_significant=r.is_significant,
                effect_size=r.effect_size,
                relative_change=r.relative_change,
                confidence_level=r.confidence_level,
                winner=r.winner.value,
            )
            for r in report.statistical_results
        ],
        recommendation=report.recommendation,
        summary=report.summary,
    )


@router.get("/{experiment_id}/assign", response_model=TrafficAssignmentResponse)
async def get_traffic_assignment(
    experiment_id: int,
    _: Annotated[None, Depends(require_permission(Permission.ANALYTICS_READ))],
    run_id: str | None = Query(None, description="Run ID for deterministic assignment"),
) -> TrafficAssignmentResponse:
    """Get traffic assignment for a run.

    Requires ANALYTICS_READ permission.
    Returns which variant a run should be assigned to based on traffic split.

    Args:
        experiment_id: Experiment ID.
        run_id: Optional run ID for deterministic assignment.

    Returns:
        TrafficAssignmentResponse with variant assignment.
    """
    manager = get_experiment_manager()

    try:
        router = manager.get_traffic_router(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    experiment = manager.get_experiment(experiment_id)
    if experiment and experiment.status != ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Experiment is not running",
        )

    # Generate run_id if not provided
    if not run_id:
        import uuid

        run_id = str(uuid.uuid4())

    variant = router.get_variant(run_id=run_id, deterministic=True)

    return TrafficAssignmentResponse(
        experiment_id=experiment_id,
        variant_name=variant.name,
        agent_name=variant.agent_name,
        run_id=run_id,
    )
