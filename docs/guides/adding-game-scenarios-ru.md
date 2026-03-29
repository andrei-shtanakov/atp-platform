# Добавление игровых сценариев для тестирования на основе теории игр

Данное руководство описывает, как добавлять новые игровые сценарии в платформу ATP. Игры находятся в отдельном пакете `game-environments` и используются модулем `atp-games` для оценки агентов.

## Обзор

Каждая игра состоит из трёх частей:

1. **Класс игры** — реализует правила, логику выплат и управление состоянием
2. **Стратегии** — базовые агенты для бенчмаркинга (например, «всегда кооперировать», «око за око»)
3. **Тесты** — проверяют корректность матриц выплат и граничные случаи

## Быстрый старт

Для добавления игры «My Game» создайте следующие файлы:

```
game-environments/
  game_envs/
    games/my_game.py                  # Игра + конфигурация
    strategies/my_game_strategies.py  # Базовые стратегии
  tests/test_my_game.py              # Тесты
```

## Шаг 1: Определите конфигурацию

Создайте файл `game-environments/game_envs/games/my_game.py`. Начните с замороженного dataclass, наследующего `GameConfig`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game_envs.core.action import DiscreteActionSpace
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import GameState, Observation, RoundResult, StepResult
from game_envs.games.registry import register_game


@dataclass(frozen=True)
class MyGameConfig(GameConfig):
    """Конфигурация для My Game.

    Ограничения: reward_high > reward_low > 0
    """

    reward_high: float = 4.0
    reward_low: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()  # ОБЯЗАТЕЛЬНО — вызывает валидацию родителя
        if not (self.reward_high > self.reward_low > 0):
            msg = "Должно выполняться: reward_high > reward_low > 0"
            raise ValueError(msg)
```

**Наследуемые поля из `GameConfig`** (доступны в любой конфигурации):
- `num_players` (по умолчанию 2) — количество игроков
- `num_rounds` (по умолчанию 1) — количество раундов
- `discount_factor` (по умолчанию 1.0) — коэффициент дисконтирования для повторных игр
- `noise` (по умолчанию 0.0) — вероятность случайной подмены действия (trembling hand)
- `seed` (необязательно) — для воспроизводимости результатов

## Шаг 2: Реализуйте игру

```python
@register_game("my_game", MyGameConfig)
class MyGame(Game):
    """My Game — описание того, что тестирует эта игра."""

    def __init__(self, config: MyGameConfig | None = None) -> None:
        super().__init__(config or MyGameConfig())
        self._terminal = False
        self._cumulative: dict[str, float] = {}

    @property
    def _cfg(self) -> MyGameConfig:
        return self.config  # type: ignore[return-value]

    # --- Обязательные свойства ---

    @property
    def name(self) -> str:
        rounds = self._cfg.num_rounds
        if rounds > 1:
            return f"My Game (повторная x{rounds})"
        return "My Game"

    @property
    def game_type(self) -> GameType:
        if self._cfg.num_rounds > 1:
            return GameType.REPEATED
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return [f"player_{i}" for i in range(self._cfg.num_players)]

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    # --- Обязательные методы ---

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        """Возвращает пространство действий для игрока."""
        return DiscreteActionSpace(["action_a", "action_b"])

    def reset(self) -> StepResult:
        """Сбрасывает игру в начальное состояние."""
        self._reset_base()  # Очищает историю и каналы сообщений
        self._terminal = False
        self._current_round = 0
        self._cumulative = {pid: 0.0 for pid in self.player_ids}
        state = GameState(
            round_number=0,
            player_states={},
            public_state={"game": self.name},
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs={pid: 0.0 for pid in self.player_ids},
            is_terminal=False,
        )

    def step(self, actions: dict[str, Any]) -> StepResult:
        """Обрабатывает один раунд действий."""
        if self._terminal:
            msg = "Игра уже завершена"
            raise RuntimeError(msg)

        p0, p1 = self.player_ids[0], self.player_ids[1]
        a0, a1 = actions[p0], actions[p1]

        # Применяем шум (trembling hand)
        if self._cfg.noise > 0:
            if self._rng.random() < self._cfg.noise:
                a0 = "action_b" if a0 == "action_a" else "action_a"
            if self._rng.random() < self._cfg.noise:
                a1 = "action_b" if a1 == "action_a" else "action_a"

        # Вычисляем выплаты (здесь ваша логика)
        if a0 == "action_a" and a1 == "action_a":
            payoffs = {p0: self._cfg.reward_high, p1: self._cfg.reward_high}
        else:
            payoffs = {p0: self._cfg.reward_low, p1: self._cfg.reward_low}

        # Применяем дисконтирование для повторных игр
        discount = self._cfg.discount_factor ** self._current_round
        for pid in self.player_ids:
            self._cumulative[pid] += payoffs[pid] * discount

        # Записываем раунд в историю
        self._history.add_round(
            RoundResult(
                round_number=self._current_round,
                actions={p0: a0, p1: a1},
                payoffs=payoffs,
            )
        )

        self._current_round += 1
        if self._current_round >= self._cfg.num_rounds:
            self._terminal = True

        state = GameState(
            round_number=self._current_round,
            player_states={},
            public_state={"game": self.name},
            is_terminal=self._terminal,
        )
        obs = {pid: self.observe(pid) for pid in self.player_ids}
        return StepResult(
            state=state,
            observations=obs,
            payoffs=payoffs,  # Выплаты ЗА РАУНД, не кумулятивные
            is_terminal=self._terminal,
        )

    def get_payoffs(self) -> dict[str, float]:
        """Возвращает кумулятивные выплаты (с учётом дисконтирования)."""
        return dict(self._cumulative)

    def observe(self, player_id: str) -> Observation:
        """Возвращает наблюдение для конкретного игрока."""
        return Observation(
            player_id=player_id,
            game_state={
                "game": self.name,
                "your_role": player_id,
            },
            available_actions=self.action_space(player_id).to_list(),
            history=self._history.for_player(player_id),
            round_number=self._current_round,
            total_rounds=self._cfg.num_rounds,
            messages=self._get_pending_messages(player_id),
        )
```

**Ключевые моменты:**
- `@register_game("my_game", MyGameConfig)` регистрирует игру в глобальном реестре
- `step()` возвращает выплаты **за раунд** в `StepResult.payoffs`, не кумулятивные
- `get_payoffs()` возвращает **кумулятивные** выплаты (с учётом discount_factor)
- `self._history` и `self._rng` предоставляются базовым классом `Game`
- `self._reset_base()` обязательно вызывается в `reset()` для очистки истории

## Шаг 3: Добавьте стратегии

Создайте `game-environments/game_envs/strategies/my_game_strategies.py`:

```python
from __future__ import annotations

from typing import Any

from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy


class AlwaysA(Strategy):
    """Всегда выбирает action_a."""

    @property
    def name(self) -> str:
        return "always_a"

    def choose_action(self, observation: Observation) -> Any:
        return "action_a"

    def reset(self) -> None:
        pass


class AlwaysB(Strategy):
    """Всегда выбирает action_b."""

    @property
    def name(self) -> str:
        return "always_b"

    def choose_action(self, observation: Observation) -> Any:
        return "action_b"

    def reset(self) -> None:
        pass


class ReactiveStrategy(Strategy):
    """Повторяет последнее действие оппонента (начинает с action_a)."""

    @property
    def name(self) -> str:
        return "reactive"

    def choose_action(self, observation: Observation) -> Any:
        if not observation.history:
            return "action_a"
        last_round = observation.history[-1]
        for pid, action in last_round.actions.items():
            if pid != observation.player_id:
                return action
        return "action_a"

    def reset(self) -> None:
        pass
```

## Шаг 4: Зарегистрируйте экспорты

Добавьте в `game-environments/game_envs/games/__init__.py`:

```python
from game_envs.games.my_game import MyGame, MyGameConfig

# Добавьте в __all__:
__all__ = [
    # ... существующие экспорты ...
    "MyGame",
    "MyGameConfig",
]
```

Добавьте в `game-environments/game_envs/strategies/__init__.py`:

```python
from game_envs.strategies.my_game_strategies import AlwaysA, AlwaysB, ReactiveStrategy

# Добавьте в __all__:
__all__ = [
    # ... существующие экспорты ...
    "AlwaysA",
    "AlwaysB",
    "ReactiveStrategy",
]
```

## Шаг 5: Напишите тесты

Создайте `game-environments/tests/test_my_game.py`:

```python
"""Тесты для реализации My Game."""

import pytest

from game_envs.games.my_game import MyGame, MyGameConfig


class TestMyGamePayoffs:
    """Проверка матрицы выплат."""

    def test_both_action_a(self) -> None:
        game = MyGame(MyGameConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "action_a", "player_1": "action_a"})
        assert result.payoffs["player_0"] == pytest.approx(4.0)
        assert result.payoffs["player_1"] == pytest.approx(4.0)

    def test_mixed_actions(self) -> None:
        game = MyGame(MyGameConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "action_a", "player_1": "action_b"})
        assert result.payoffs["player_0"] == pytest.approx(1.0)

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError):
            MyGameConfig(reward_high=1.0, reward_low=5.0)


class TestMyGameRepeated:
    """Тесты для повторных игр."""

    def test_multi_round_accumulates(self) -> None:
        game = MyGame(MyGameConfig(num_players=2, num_rounds=3))
        game.reset()
        for _ in range(3):
            game.step({"player_0": "action_a", "player_1": "action_a"})
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(12.0)  # 3 * 4.0

    def test_single_round_terminal(self) -> None:
        game = MyGame(MyGameConfig(num_players=2, num_rounds=1))
        game.reset()
        result = game.step({"player_0": "action_a", "player_1": "action_a"})
        assert result.is_terminal


class TestMyGameRegistry:
    """Тест интеграции с реестром."""

    def test_in_registry(self) -> None:
        from game_envs.games.registry import GameRegistry

        assert "my_game" in GameRegistry.list_games()
```

## Шаг 6: Обновите счётчик в тесте реестра

В файле `game-environments/tests/test_registry.py` обновите проверку количества игр:

```python
# Найдите строку: assert len(result) == 7
# Измените на:     assert len(result) == 8
```

## Шаг 7: Запустите тесты и проверку качества

```bash
# Запуск ваших тестов
uv run python -m pytest game-environments/tests/test_my_game.py -v

# Запуск всех тестов game-environments
uv run python -m pytest game-environments/tests/ -v

# Проверка качества кода
uv run ruff format game-environments/
uv run ruff check game-environments/ --fix
```

## Типы пространств действий

Выберите подходящее пространство действий для вашей игры:

| Тип | Применение | Пример |
|-----|-----------|--------|
| `DiscreteActionSpace(["a", "b"])` | Конечный набор именованных действий | Кооперация/предательство, камень/ножницы/бумага |
| `ContinuousActionSpace(0.0, 100.0)` | Числовой диапазон | Размер ставки, уровень вклада |
| `StructuredActionSpace(schema)` | Вектор распределения | Размещение войск, распределение ресурсов |

## Существующие игры как образцы

| Игра | Файл | Тип действий | Игроки | Особенность |
|------|------|-------------|--------|-------------|
| Дилемма заключённого | `prisoners_dilemma.py` | Дискретный | 2 | Кооперация vs предательство |
| Охота на оленя | `stag_hunt.py` | Дискретный | 2 | Доверие vs безопасность (**лучший шаблон**) |
| Битва полов | `battle_of_sexes.py` | Дискретный | 2 | Асимметричные выплаты |
| Общественное благо | `public_goods.py` | Непрерывный | 2-20 | N игроков, суммы вкладов |
| Аукцион | `auction.py` | Непрерывный | 2+ | Частные оценки, частичная наблюдаемость |
| Полковник Блотто | `colonel_blotto.py` | Структурированный | 2 | Распределение с ограничением на сумму |
| Игра заторов | `congestion.py` | Дискретный | 2-50 | Выбор маршрута, отрицательные выплаты |

**Рекомендуемый шаблон для старта:** `stag_hunt.py` — самая простая дискретная игра для двух игроков.

## Чек-лист

- [ ] Конфигурация (`@dataclass(frozen=True)`, наследует `GameConfig`, валидация в `__post_init__`)
- [ ] Класс игры (`@register_game`, все абстрактные методы реализованы)
- [ ] Стратегии (минимум 2-3 базовых, каждая наследует `Strategy`)
- [ ] Экспорты в `games/__init__.py` и `strategies/__init__.py`
- [ ] Тесты: корректность выплат, валидация конфигурации, повторные раунды, терминальное состояние, реестр
- [ ] Счётчик в тесте реестра обновлён
- [ ] `ruff format` и `ruff check` проходят
