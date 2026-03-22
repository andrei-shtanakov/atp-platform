"""Unit tests for A/B Testing Framework."""

import pytest

from atp.analytics.ab_testing import (
    Experiment,
    ExperimentAnalyzer,
    ExperimentConfig,
    ExperimentManager,
    ExperimentObservation,
    ExperimentStatus,
    MetricConfig,
    MetricType,
    RollbackConfig,
    TrafficRouter,
    Variant,
    VariantType,
    WinnerDecision,
    get_experiment_manager,
    reset_experiment_manager,
)

# ==================== Fixtures ====================


@pytest.fixture
def basic_config() -> ExperimentConfig:
    """Create a basic experiment configuration."""
    return ExperimentConfig(
        name="test-experiment",
        description="Test experiment",
        suite_name="test-suite.yaml",
        control_variant=Variant(
            name="control",
            variant_type=VariantType.CONTROL,
            agent_name="agent-v1",
            traffic_weight=50.0,
        ),
        treatment_variant=Variant(
            name="treatment",
            variant_type=VariantType.TREATMENT,
            agent_name="agent-v2",
            traffic_weight=50.0,
        ),
        metrics=[
            MetricConfig(
                metric_type=MetricType.SCORE,
                is_primary=True,
                minimize=False,
            )
        ],
        min_sample_size=10,
    )


@pytest.fixture
def manager() -> ExperimentManager:
    """Create a fresh experiment manager."""
    reset_experiment_manager()
    return ExperimentManager()


@pytest.fixture
def analyzer() -> ExperimentAnalyzer:
    """Create an experiment analyzer."""
    return ExperimentAnalyzer()


@pytest.fixture
def running_experiment(
    manager: ExperimentManager,
    basic_config: ExperimentConfig,
) -> Experiment:
    """Create and start an experiment."""
    experiment = manager.create_experiment(basic_config)
    manager.start_experiment(experiment.id)
    return experiment


# ==================== Variant Tests ====================


class TestVariant:
    """Tests for Variant model."""

    def test_variant_creation(self) -> None:
        """Test creating a variant."""
        variant = Variant(
            name="control",
            variant_type=VariantType.CONTROL,
            agent_name="agent-v1",
            traffic_weight=60.0,
            description="Control variant",
        )

        assert variant.name == "control"
        assert variant.variant_type == VariantType.CONTROL
        assert variant.agent_name == "agent-v1"
        assert variant.traffic_weight == 60.0
        assert variant.description == "Control variant"

    def test_variant_default_weight(self) -> None:
        """Test variant default traffic weight."""
        variant = Variant(
            name="treatment",
            variant_type=VariantType.TREATMENT,
            agent_name="agent-v2",
        )

        assert variant.traffic_weight == 50.0


# ==================== ExperimentConfig Tests ====================


class TestExperimentConfig:
    """Tests for ExperimentConfig model."""

    def test_config_creation(self, basic_config: ExperimentConfig) -> None:
        """Test creating an experiment config."""
        assert basic_config.name == "test-experiment"
        assert basic_config.suite_name == "test-suite.yaml"
        assert basic_config.control_variant.name == "control"
        assert basic_config.treatment_variant.name == "treatment"
        assert len(basic_config.metrics) == 1
        assert basic_config.metrics[0].is_primary is True

    def test_config_default_values(self) -> None:
        """Test config default values."""
        config = ExperimentConfig(
            name="test",
            suite_name="suite.yaml",
            control_variant=Variant(
                name="control",
                variant_type=VariantType.CONTROL,
                agent_name="agent-a",
            ),
            treatment_variant=Variant(
                name="treatment",
                variant_type=VariantType.TREATMENT,
                agent_name="agent-b",
            ),
        )

        assert config.min_sample_size == 30
        assert config.max_sample_size is None
        assert config.significance_level == 0.05
        assert config.rollback.enabled is True


# ==================== Experiment Tests ====================


class TestExperiment:
    """Tests for Experiment model."""

    def test_experiment_creation(self, basic_config: ExperimentConfig) -> None:
        """Test creating an experiment."""
        experiment = Experiment(config=basic_config)

        assert experiment.status == ExperimentStatus.DRAFT
        assert experiment.id is None
        assert experiment.control_sample_size == 0
        assert experiment.treatment_sample_size == 0
        assert experiment.is_active is False

    def test_experiment_sample_size_properties(
        self, basic_config: ExperimentConfig
    ) -> None:
        """Test sample size property calculations."""
        experiment = Experiment(config=basic_config)

        # Add some observations
        for i in range(5):
            experiment.control_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    score=80.0,
                )
            )
        for i in range(3):
            experiment.treatment_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    score=85.0,
                )
            )

        assert experiment.control_sample_size == 5
        assert experiment.treatment_sample_size == 3
        assert experiment.total_sample_size == 8

    def test_can_conclude(self, basic_config: ExperimentConfig) -> None:
        """Test can_conclude property."""
        basic_config.min_sample_size = 3
        experiment = Experiment(config=basic_config)

        assert experiment.can_conclude is False

        # Add minimum samples
        for i in range(3):
            experiment.control_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    score=80.0,
                )
            )
            experiment.treatment_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    score=85.0,
                )
            )

        assert experiment.can_conclude is True


# ==================== TrafficRouter Tests ====================


class TestTrafficRouter:
    """Tests for TrafficRouter."""

    def test_equal_split(self, basic_config: ExperimentConfig) -> None:
        """Test 50/50 traffic split."""
        experiment = Experiment(id=1, config=basic_config)
        router = TrafficRouter(experiment)

        # With deterministic assignment, same run_id should always get same variant
        variant1 = router.get_variant(run_id="test-run-1", deterministic=True)
        variant2 = router.get_variant(run_id="test-run-1", deterministic=True)
        assert variant1.name == variant2.name

    def test_deterministic_assignment(self, basic_config: ExperimentConfig) -> None:
        """Test that same run_id always gets same variant."""
        experiment = Experiment(id=1, config=basic_config)
        router = TrafficRouter(experiment)

        # Run multiple times with same run_id
        assignments = [
            router.get_variant(run_id="consistent-run", deterministic=True).name
            for _ in range(10)
        ]

        # All assignments should be the same
        assert len(set(assignments)) == 1

    def test_different_run_ids_vary(self, basic_config: ExperimentConfig) -> None:
        """Test that different run_ids get different variants over many runs."""
        experiment = Experiment(id=1, config=basic_config)
        router = TrafficRouter(experiment)

        control_count = 0
        treatment_count = 0

        # With many different run_ids, we should see both variants
        for i in range(100):
            variant = router.get_variant(run_id=f"run-{i}", deterministic=True)
            if variant.name == "control":
                control_count += 1
            else:
                treatment_count += 1

        # Both variants should have been assigned (with 50/50 split)
        assert control_count > 0
        assert treatment_count > 0

    def test_skewed_split(self) -> None:
        """Test non-equal traffic split."""
        config = ExperimentConfig(
            name="skewed-test",
            suite_name="suite.yaml",
            control_variant=Variant(
                name="control",
                variant_type=VariantType.CONTROL,
                agent_name="agent-a",
                traffic_weight=90.0,
            ),
            treatment_variant=Variant(
                name="treatment",
                variant_type=VariantType.TREATMENT,
                agent_name="agent-b",
                traffic_weight=10.0,
            ),
        )
        experiment = Experiment(id=1, config=config)
        router = TrafficRouter(experiment)

        control_count = 0
        treatment_count = 0

        for i in range(1000):
            variant = router.get_variant(run_id=f"run-{i}", deterministic=True)
            if variant.name == "control":
                control_count += 1
            else:
                treatment_count += 1

        # Control should get significantly more traffic
        assert control_count > treatment_count * 5  # Roughly 90/10


# ==================== ExperimentAnalyzer Tests ====================


class TestExperimentAnalyzer:
    """Tests for ExperimentAnalyzer."""

    def test_compute_variant_metrics_empty(self, analyzer: ExperimentAnalyzer) -> None:
        """Test computing metrics with empty observations."""
        metrics = analyzer.compute_variant_metrics([], MetricType.SCORE)

        assert metrics.sample_size == 0
        assert metrics.mean == 0.0
        assert metrics.std == 0.0

    def test_compute_variant_metrics(self, analyzer: ExperimentAnalyzer) -> None:
        """Test computing metrics with observations."""
        observations = [
            ExperimentObservation(
                experiment_id=1,
                variant_name="control",
                test_id="test-1",
                run_id=f"run-{i}",
                score=80.0 + i,
            )
            for i in range(5)
        ]

        metrics = analyzer.compute_variant_metrics(observations, MetricType.SCORE)

        assert metrics.sample_size == 5
        assert metrics.mean == 82.0  # (80+81+82+83+84)/5
        assert metrics.min_value == 80.0
        assert metrics.max_value == 84.0
        assert metrics.ci_lower is not None
        assert metrics.ci_upper is not None

    def test_compute_statistical_result(
        self,
        analyzer: ExperimentAnalyzer,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test computing statistical significance."""
        basic_config.min_sample_size = 5
        experiment = Experiment(id=1, config=basic_config)

        # Add control observations (mean ~80)
        for i in range(10):
            experiment.control_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    score=80.0 + (i % 5),
                )
            )

        # Add treatment observations (mean ~90 - significant improvement)
        for i in range(10):
            experiment.treatment_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    score=90.0 + (i % 5),
                )
            )

        result = analyzer.compute_statistical_result(
            experiment, basic_config.metrics[0]
        )

        assert result.control_metrics.sample_size == 10
        assert result.treatment_metrics.sample_size == 10
        assert result.relative_change > 0  # Treatment is better
        assert result.p_value < 1.0

    def test_check_for_degradation_insufficient_samples(
        self,
        analyzer: ExperimentAnalyzer,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test degradation check with insufficient samples."""
        experiment = Experiment(id=1, config=basic_config)

        # Add a few samples (below min threshold)
        for i in range(5):
            experiment.control_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    score=80.0,
                )
            )
            experiment.treatment_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    score=70.0,
                )
            )

        is_degraded, reason = analyzer.check_for_degradation(experiment)

        # Should not trigger with insufficient samples
        assert is_degraded is False
        assert reason is None

    def test_check_for_degradation_detected(
        self,
        analyzer: ExperimentAnalyzer,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test degradation detection when treatment is significantly worse."""
        basic_config.rollback = RollbackConfig(
            enabled=True,
            degradation_threshold=0.05,  # 5% threshold
            min_samples_before_rollback=10,
        )
        experiment = Experiment(id=1, config=basic_config)

        # Add samples with treatment significantly worse
        for i in range(30):
            experiment.control_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    score=80.0,
                )
            )
            experiment.treatment_observations.append(
                ExperimentObservation(
                    experiment_id=1,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    score=70.0,  # 12.5% worse
                )
            )

        is_degraded, reason = analyzer.check_for_degradation(experiment)

        assert is_degraded is True
        assert reason is not None
        assert "decrease" in reason.lower() or "12.5%" in reason


# ==================== ExperimentManager Tests ====================


class TestExperimentManager:
    """Tests for ExperimentManager."""

    def test_create_experiment(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test creating an experiment."""
        experiment = manager.create_experiment(basic_config)

        assert experiment.id is not None
        assert experiment.id == 1
        assert experiment.status == ExperimentStatus.DRAFT
        assert experiment.config.name == "test-experiment"

    def test_get_experiment(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test retrieving an experiment."""
        created = manager.create_experiment(basic_config)
        retrieved = manager.get_experiment(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.config.name == created.config.name

    def test_get_nonexistent_experiment(self, manager: ExperimentManager) -> None:
        """Test retrieving a nonexistent experiment."""
        experiment = manager.get_experiment(999)
        assert experiment is None

    def test_list_experiments(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test listing experiments."""
        # Create multiple experiments
        manager.create_experiment(basic_config)

        basic_config2 = basic_config.model_copy(deep=True)
        basic_config2.name = "test-experiment-2"
        manager.create_experiment(basic_config2)

        experiments = manager.list_experiments()

        assert len(experiments) == 2

    def test_list_experiments_with_filter(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test listing experiments with status filter."""
        exp1 = manager.create_experiment(basic_config)
        manager.start_experiment(exp1.id)

        basic_config2 = basic_config.model_copy(deep=True)
        basic_config2.name = "test-experiment-2"
        manager.create_experiment(basic_config2)  # Stays in draft

        running = manager.list_experiments(status=ExperimentStatus.RUNNING)
        draft = manager.list_experiments(status=ExperimentStatus.DRAFT)

        assert len(running) == 1
        assert len(draft) == 1

    def test_start_experiment(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test starting an experiment."""
        experiment = manager.create_experiment(basic_config)
        started = manager.start_experiment(experiment.id)

        assert started.status == ExperimentStatus.RUNNING
        assert started.started_at is not None

    def test_start_experiment_already_running(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test starting an already running experiment."""
        experiment = manager.create_experiment(basic_config)
        manager.start_experiment(experiment.id)

        with pytest.raises(ValueError, match="Cannot start"):
            manager.start_experiment(experiment.id)

    def test_pause_experiment(
        self,
        manager: ExperimentManager,
        running_experiment: Experiment,
    ) -> None:
        """Test pausing an experiment."""
        paused = manager.pause_experiment(running_experiment.id)

        assert paused.status == ExperimentStatus.PAUSED
        assert paused.paused_at is not None

    def test_resume_experiment(
        self,
        manager: ExperimentManager,
        running_experiment: Experiment,
    ) -> None:
        """Test resuming a paused experiment."""
        manager.pause_experiment(running_experiment.id)
        resumed = manager.resume_experiment(running_experiment.id)

        assert resumed.status == ExperimentStatus.RUNNING
        assert resumed.paused_at is None

    def test_conclude_experiment(
        self,
        manager: ExperimentManager,
        running_experiment: Experiment,
    ) -> None:
        """Test concluding an experiment."""
        concluded = manager.conclude_experiment(
            running_experiment.id, "Test conclusion"
        )

        assert concluded.status == ExperimentStatus.CONCLUDED
        assert concluded.concluded_at is not None
        assert concluded.conclusion_reason == "Test conclusion"

    def test_record_observation(
        self,
        manager: ExperimentManager,
        running_experiment: Experiment,
    ) -> None:
        """Test recording an observation."""
        observation = ExperimentObservation(
            experiment_id=running_experiment.id,
            variant_name="control",
            test_id="test-1",
            run_id="run-1",
            score=85.0,
            success=True,
        )

        experiment = manager.record_observation(running_experiment.id, observation)

        assert experiment.control_sample_size == 1
        assert experiment.control_observations[0].score == 85.0

    def test_record_observation_triggers_rollback(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test that degradation triggers rollback."""
        basic_config.rollback = RollbackConfig(
            enabled=True,
            degradation_threshold=0.05,
            min_samples_before_rollback=10,
            consecutive_checks=1,  # Immediate rollback
        )
        experiment = manager.create_experiment(basic_config)
        manager.start_experiment(experiment.id)

        # Add observations with treatment significantly worse
        # Stop once we detect rollback
        for i in range(30):
            exp = manager.get_experiment(experiment.id)
            if exp.status == ExperimentStatus.ROLLED_BACK:
                break

            control_obs = ExperimentObservation(
                experiment_id=experiment.id,
                variant_name="control",
                test_id="test-1",
                run_id=f"run-control-{i}",
                score=80.0,
            )
            manager.record_observation(experiment.id, control_obs)

            # Check again after control observation
            exp = manager.get_experiment(experiment.id)
            if exp.status == ExperimentStatus.ROLLED_BACK:
                break

            treatment_obs = ExperimentObservation(
                experiment_id=experiment.id,
                variant_name="treatment",
                test_id="test-1",
                run_id=f"run-treatment-{i}",
                score=60.0,  # 25% worse
            )
            manager.record_observation(experiment.id, treatment_obs)

        # Check if rollback was triggered
        experiment = manager.get_experiment(experiment.id)
        assert experiment.rollback_triggered is True
        assert experiment.status == ExperimentStatus.ROLLED_BACK
        assert experiment.winner == WinnerDecision.CONTROL

    def test_get_traffic_router(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test getting a traffic router."""
        experiment = manager.create_experiment(basic_config)
        router = manager.get_traffic_router(experiment.id)

        assert router is not None
        variant = router.get_variant(run_id="test", deterministic=True)
        assert variant.name in ["control", "treatment"]

    def test_generate_report(
        self,
        manager: ExperimentManager,
        running_experiment: Experiment,
    ) -> None:
        """Test generating an experiment report."""
        # Add some observations
        for i in range(10):
            manager.record_observation(
                running_experiment.id,
                ExperimentObservation(
                    experiment_id=running_experiment.id,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    score=80.0,
                ),
            )
            manager.record_observation(
                running_experiment.id,
                ExperimentObservation(
                    experiment_id=running_experiment.id,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    score=85.0,
                ),
            )

        report = manager.generate_report(running_experiment.id)

        assert report.experiment.id == running_experiment.id
        assert len(report.statistical_results) > 0
        assert report.recommendation != ""
        assert "experiment_name" in report.summary


# ==================== Module Functions Tests ====================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_experiment_manager_singleton(self) -> None:
        """Test that get_experiment_manager returns singleton."""
        reset_experiment_manager()
        manager1 = get_experiment_manager()
        manager2 = get_experiment_manager()

        assert manager1 is manager2

    def test_reset_experiment_manager(self) -> None:
        """Test resetting the experiment manager."""
        manager1 = get_experiment_manager()
        reset_experiment_manager()
        manager2 = get_experiment_manager()

        assert manager1 is not manager2


# ==================== Winner Determination Tests ====================


class TestWinnerDetermination:
    """Tests for winner determination logic."""

    def test_treatment_wins_higher_score(
        self,
        manager: ExperimentManager,
        basic_config: ExperimentConfig,
    ) -> None:
        """Test that treatment wins with significantly higher score."""
        basic_config.min_sample_size = 5
        experiment = manager.create_experiment(basic_config)
        manager.start_experiment(experiment.id)

        # Add observations with treatment clearly better
        for i in range(20):
            manager.record_observation(
                experiment.id,
                ExperimentObservation(
                    experiment_id=experiment.id,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    score=70.0 + (i % 5),
                ),
            )
            manager.record_observation(
                experiment.id,
                ExperimentObservation(
                    experiment_id=experiment.id,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    score=90.0 + (i % 5),
                ),
            )

        manager.conclude_experiment(experiment.id, "Test complete")
        experiment = manager.get_experiment(experiment.id)

        assert experiment.winner == WinnerDecision.TREATMENT

    def test_control_wins_for_minimize_metric(
        self,
        manager: ExperimentManager,
    ) -> None:
        """Test that control wins when minimizing metric and treatment is higher."""
        config = ExperimentConfig(
            name="minimize-test",
            suite_name="suite.yaml",
            control_variant=Variant(
                name="control",
                variant_type=VariantType.CONTROL,
                agent_name="agent-a",
            ),
            treatment_variant=Variant(
                name="treatment",
                variant_type=VariantType.TREATMENT,
                agent_name="agent-b",
            ),
            metrics=[
                MetricConfig(
                    metric_type=MetricType.DURATION,
                    is_primary=True,
                    minimize=True,  # Lower is better
                )
            ],
            min_sample_size=10,  # Must be >= 10 per validation
        )

        experiment = manager.create_experiment(config)
        manager.start_experiment(experiment.id)

        # Control has lower duration (better)
        for i in range(20):
            manager.record_observation(
                experiment.id,
                ExperimentObservation(
                    experiment_id=experiment.id,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    duration_seconds=10.0 + (i % 5),
                ),
            )
            manager.record_observation(
                experiment.id,
                ExperimentObservation(
                    experiment_id=experiment.id,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    duration_seconds=30.0 + (i % 5),
                ),
            )

        manager.conclude_experiment(experiment.id, "Test complete")
        experiment = manager.get_experiment(experiment.id)

        assert experiment.winner == WinnerDecision.CONTROL

    def test_small_effect_size_results_in_tie(
        self,
        manager: ExperimentManager,
    ) -> None:
        """Test that small effect size results in tie even with significance."""
        # Create config requiring large effect size to declare winner
        config = ExperimentConfig(
            name="tie-test",
            suite_name="suite.yaml",
            control_variant=Variant(
                name="control",
                variant_type=VariantType.CONTROL,
                agent_name="agent-a",
            ),
            treatment_variant=Variant(
                name="treatment",
                variant_type=VariantType.TREATMENT,
                agent_name="agent-b",
            ),
            metrics=[
                MetricConfig(
                    metric_type=MetricType.SCORE,
                    is_primary=True,
                    min_effect_size=0.8,  # Require large effect to declare winner
                )
            ],
            min_sample_size=10,
        )
        experiment = manager.create_experiment(config)
        manager.start_experiment(experiment.id)

        # Both variants perform with small difference (effect size < 0.8)
        for i in range(20):
            manager.record_observation(
                experiment.id,
                ExperimentObservation(
                    experiment_id=experiment.id,
                    variant_name="control",
                    test_id="test-1",
                    run_id=f"run-control-{i}",
                    score=80.0 + (i % 5),  # Mean = 82
                ),
            )
            manager.record_observation(
                experiment.id,
                ExperimentObservation(
                    experiment_id=experiment.id,
                    variant_name="treatment",
                    test_id="test-1",
                    run_id=f"run-treatment-{i}",
                    score=81.0 + (i % 5),  # Mean = 83, small difference
                ),
            )

        manager.conclude_experiment(experiment.id, "Test complete")
        experiment = manager.get_experiment(experiment.id)

        # With high min_effect_size, small difference should result in tie
        assert experiment.winner == WinnerDecision.TIE
