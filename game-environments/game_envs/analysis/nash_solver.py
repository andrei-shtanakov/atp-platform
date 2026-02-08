"""Nash equilibrium solvers for game-theoretic analysis.

Provides four algorithms:
- Support enumeration: exact, finds all NE for 2-player games
- Lemke-Howson: efficient single NE for 2-player games
- Fictitious play: approximate NE for n-player games
- Replicator dynamics: evolutionary equilibrium
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from game_envs.analysis.models import NashEquilibrium


class NashSolver:
    """Collection of Nash equilibrium solvers."""

    @staticmethod
    def support_enumeration(
        payoff_1: np.ndarray,
        payoff_2: np.ndarray,
        tol: float = 1e-10,
    ) -> list[NashEquilibrium]:
        """Find all Nash equilibria via support enumeration.

        Enumerates support pairs where both supports have equal
        size (non-degenerate condition). For each valid pair,
        solves the indifference equations and checks best
        response conditions.

        Args:
            payoff_1: (m x n) payoff matrix for player 1.
            payoff_2: (m x n) payoff matrix for player 2.
            tol: Numerical tolerance for zero checks.

        Returns:
            List of all Nash equilibria found.
        """
        m, n = payoff_1.shape
        equilibria: list[NashEquilibrium] = []

        # Non-degenerate NE have equal support sizes
        # Pre-compute sub-matrices for reuse
        for k in range(1, min(m, n) + 1):
            s1_list = list(combinations(range(m), k))
            s2_list = list(combinations(range(n), k))
            for s1 in s1_list:
                for s2 in s2_list:
                    result = _solve_support(
                        payoff_1,
                        payoff_2,
                        list(s1),
                        list(s2),
                        tol,
                    )
                    if result is not None:
                        equilibria.append(result)

        return _deduplicate(equilibria, tol)

    @staticmethod
    def lemke_howson(
        payoff_1: np.ndarray,
        payoff_2: np.ndarray,
        initial_label: int = 0,
        max_iterations: int = 1000,
    ) -> NashEquilibrium:
        """Find one Nash equilibrium via Lemke-Howson algorithm.

        Efficient complementary pivoting algorithm for 2-player
        bimatrix games. Follows a path of complementary pivots
        from an artificial equilibrium to a real one.

        Args:
            payoff_1: (m x n) payoff matrix for player 1.
            payoff_2: (m x n) payoff matrix for player 2.
            initial_label: Starting label for pivoting (0..m+n-1).
            max_iterations: Maximum pivot steps.

        Returns:
            A single Nash equilibrium.
        """
        m, n = payoff_1.shape

        # Shift payoffs to be strictly positive (required)
        shift = max(
            -payoff_1.min() + 1,
            -payoff_2.min() + 1,
            1.0,
        )
        a = payoff_2.T + shift  # n x m (column player's payoffs)
        b = payoff_1 + shift  # m x n (row player's payoffs)

        # Build tableaux
        # Player 1 tableau: [I_m | B | 1_m]
        # Player 2 tableau: [A | I_n | 1_n]
        tab1 = np.hstack(
            [
                np.eye(m),
                b,
                np.ones((m, 1)),
            ]
        )
        tab2 = np.hstack(
            [
                a,
                np.eye(n),
                np.ones((n, 1)),
            ]
        )

        # Basic variables for each tableau
        # Player 1 basics: labels 0..m-1 initially
        # Player 2 basics: labels m..m+n-1 initially
        basic1 = list(range(m))
        basic2 = list(range(m, m + n))

        # Drop initial_label to start pivoting
        entering = initial_label

        for _ in range(max_iterations):
            if entering < m:
                # Pivot in player 2's tableau
                col = entering
                ratios = tab2[:, m + n] / np.maximum(
                    tab2[:, col],
                    1e-15,
                )
                # Only consider positive entries
                mask = tab2[:, col] > 1e-15
                if not mask.any():
                    break
                ratios[~mask] = np.inf
                pivot_row = int(np.argmin(ratios))

                # Perform pivot
                pivot_val = tab2[pivot_row, col]
                tab2[pivot_row] /= pivot_val
                for i in range(n):
                    if i != pivot_row:
                        tab2[i] -= tab2[i, col] * tab2[pivot_row]

                leaving = basic2[pivot_row]
                basic2[pivot_row] = entering
                entering = leaving
            else:
                # Pivot in player 1's tableau
                # Labels m..m+n-1 map to columns m..m+n-1 in tab1
                # tab1 has columns: [I_m | B | 1] where I_m is 0..m-1, B is m..m+n-1
                tab_col = entering

                ratios = tab1[:, m + n] / np.maximum(
                    tab1[:, tab_col],
                    1e-15,
                )
                mask = tab1[:, tab_col] > 1e-15
                if not mask.any():
                    break
                ratios[~mask] = np.inf
                pivot_row = int(np.argmin(ratios))

                pivot_val = tab1[pivot_row, tab_col]
                tab1[pivot_row] /= pivot_val
                for i in range(m):
                    if i != pivot_row:
                        tab1[i] -= tab1[i, tab_col] * tab1[pivot_row]

                leaving = basic1[pivot_row]
                basic1[pivot_row] = entering
                entering = leaving

            # Check if we've completed the path
            if entering == initial_label:
                break

        # Extract strategies from tableaux
        x = np.zeros(m)  # Player 1's strategy
        y = np.zeros(n)  # Player 2's strategy

        for i, label in enumerate(basic2):
            if label < m:
                x[label] = tab2[i, m + n]

        for i, label in enumerate(basic1):
            if label >= m:
                y[label - m] = tab1[i, m + n]

        # Normalize to probability distributions
        x_sum = x.sum()
        y_sum = y.sum()
        if x_sum > 1e-15:
            x /= x_sum
        else:
            x = np.ones(m) / m
        if y_sum > 1e-15:
            y /= y_sum
        else:
            y = np.ones(n) / n

        # Compute expected payoffs
        payoff_p1 = float(x @ payoff_1 @ y)
        payoff_p2 = float(x @ payoff_2 @ y)

        support_1 = [i for i in range(m) if x[i] > 1e-10]
        support_2 = [j for j in range(n) if y[j] > 1e-10]

        return NashEquilibrium(
            strategies={"player_0": x, "player_1": y},
            payoffs={"player_0": payoff_p1, "player_1": payoff_p2},
            support={"player_0": support_1, "player_1": support_2},
        )

    @staticmethod
    def fictitious_play(
        payoff_tensors: list[np.ndarray],
        max_iterations: int = 10000,
        epsilon: float = 0.01,
    ) -> NashEquilibrium:
        """Find approximate NE via fictitious play.

        Each player best-responds to the empirical frequency of
        opponents' past actions. Converges to NE in many games.

        Args:
            payoff_tensors: List of payoff arrays, one per
                player. Each has shape (a_0, a_1, ..., a_{n-1})
                where a_i is the number of actions for player i.
            max_iterations: Maximum number of iterations.
            epsilon: Convergence tolerance for strategy change.

        Returns:
            Approximate Nash equilibrium.
        """
        num_players = len(payoff_tensors)
        shape = payoff_tensors[0].shape
        num_actions = list(shape)

        # Initialize action counts uniformly
        counts: list[np.ndarray] = [np.ones(num_actions[i]) for i in range(num_players)]
        max_change = float("inf")

        for iteration in range(1, max_iterations + 1):
            old_strategies = [c / c.sum() for c in counts]

            for player in range(num_players):
                # Compute expected payoff for each action
                expected = _expected_payoffs(
                    payoff_tensors[player],
                    old_strategies,
                    player,
                )
                best_action = int(np.argmax(expected))
                counts[player][best_action] += 1

            # Check convergence
            new_strategies = [c / c.sum() for c in counts]
            max_change = max(
                float(np.max(np.abs(new - old)))
                for new, old in zip(
                    new_strategies,
                    old_strategies,
                )
            )
            if max_change < epsilon:
                break

        strategies = {f"player_{i}": c / c.sum() for i, c in enumerate(counts)}

        # Compute expected payoffs at the equilibrium
        final_strats = [s for s in strategies.values()]
        payoffs = {}
        for i in range(num_players):
            ep = _expected_payoffs(
                payoff_tensors[i],
                final_strats,
                i,
            )
            payoffs[f"player_{i}"] = float(np.dot(final_strats[i], ep))

        support = {
            pid: [j for j in range(len(s)) if s[j] > 1e-10]
            for pid, s in strategies.items()
        }

        return NashEquilibrium(
            strategies=strategies,
            payoffs=payoffs,
            support=support,
            epsilon=max_change,
        )

    @staticmethod
    def replicator_dynamics(
        payoff_tensors: list[np.ndarray],
        dt: float = 0.01,
        max_steps: int = 10000,
        epsilon: float = 1e-6,
        initial: list[np.ndarray] | None = None,
    ) -> NashEquilibrium:
        """Find evolutionary equilibrium via replicator dynamics.

        Simulates the continuous-time replicator equation where
        strategies grow in proportion to their fitness advantage
        over the population average.

        Args:
            payoff_tensors: Payoff arrays per player.
            dt: Time step for Euler integration.
            max_steps: Maximum integration steps.
            epsilon: Convergence tolerance.
            initial: Optional initial strategy profile.

        Returns:
            Nash equilibrium found by replicator dynamics.
        """
        num_players = len(payoff_tensors)
        shape = payoff_tensors[0].shape
        num_actions = list(shape)

        # Initialize strategies (slightly perturbed uniform)
        if initial is not None:
            strategies = [s.copy() for s in initial]
        else:
            rng = np.random.default_rng(42)
            strategies = []
            for i in range(num_players):
                s = rng.dirichlet(
                    np.ones(num_actions[i]) * 10,
                )
                strategies.append(s)

        max_change = epsilon
        for _ in range(max_steps):
            new_strategies = []
            for player in range(num_players):
                expected = _expected_payoffs(
                    payoff_tensors[player],
                    strategies,
                    player,
                )
                avg_fitness = float(np.dot(strategies[player], expected))

                # Replicator equation: dx_i/dt = x_i * (f_i - avg)
                delta = strategies[player] * (expected - avg_fitness)
                new_s = strategies[player] + dt * delta

                # Project onto simplex (clip + renormalize)
                new_s = np.maximum(new_s, 0.0)
                s_sum = new_s.sum()
                if s_sum > 1e-15:
                    new_s /= s_sum
                else:
                    new_s = np.ones(num_actions[player])
                    new_s /= new_s.sum()
                new_strategies.append(new_s)

            max_change = max(
                float(np.max(np.abs(new - old)))
                for new, old in zip(new_strategies, strategies)
            )
            strategies = new_strategies

            if max_change < epsilon:
                break

        result_strategies = {f"player_{i}": s for i, s in enumerate(strategies)}

        # Compute payoffs
        payoffs: dict[str, float] = {}
        for i in range(num_players):
            ep = _expected_payoffs(
                payoff_tensors[i],
                strategies,
                i,
            )
            payoffs[f"player_{i}"] = float(np.dot(strategies[i], ep))

        support = {
            pid: [j for j in range(len(s)) if s[j] > 1e-10]
            for pid, s in result_strategies.items()
        }

        return NashEquilibrium(
            strategies=result_strategies,
            payoffs=payoffs,
            support=support,
            epsilon=max_change,
        )

    @staticmethod
    def solve_2player(
        payoff_1: np.ndarray,
        payoff_2: np.ndarray,
        method: str = "support_enumeration",
    ) -> list[NashEquilibrium]:
        """Solve a 2-player bimatrix game.

        Convenience method that dispatches to the appropriate
        solver based on the method parameter.

        For large games (> 7x7), defaults to lemke_howson
        when support_enumeration is requested, as the
        combinatorial explosion makes enumeration impractical.

        Args:
            payoff_1: (m x n) payoff matrix for player 1.
            payoff_2: (m x n) payoff matrix for player 2.
            method: One of "support_enumeration" or
                "lemke_howson".

        Returns:
            List of Nash equilibria found.
        """
        if method == "support_enumeration":
            return NashSolver.support_enumeration(
                payoff_1,
                payoff_2,
            )
        if method == "lemke_howson":
            return [NashSolver.lemke_howson(payoff_1, payoff_2)]
        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def solve_nplayer(
        payoff_tensors: list[np.ndarray],
        method: str = "fictitious_play",
        max_iterations: int = 10000,
        epsilon: float = 0.01,
    ) -> NashEquilibrium:
        """Solve an n-player game.

        Args:
            payoff_tensors: Payoff arrays per player.
            method: One of "fictitious_play" or
                "replicator_dynamics".
            max_iterations: Maximum iterations.
            epsilon: Convergence tolerance.

        Returns:
            Approximate Nash equilibrium.
        """
        if method == "fictitious_play":
            return NashSolver.fictitious_play(
                payoff_tensors,
                max_iterations,
                epsilon,
            )
        if method == "replicator_dynamics":
            return NashSolver.replicator_dynamics(
                payoff_tensors,
                max_steps=max_iterations,
                epsilon=epsilon,
            )
        raise ValueError(f"Unknown method: {method}")


def _expected_payoffs(
    payoff_tensor: np.ndarray,
    strategies: list[np.ndarray],
    player: int,
) -> np.ndarray:
    """Compute expected payoffs for each action of a player.

    Marginalizes over opponents' mixed strategies to get
    the expected payoff for each pure action of the player.

    Args:
        payoff_tensor: Payoff array with shape matching the
            action space dimensions of all players.
        strategies: Current mixed strategy for each player.
        player: Index of the player to compute for.

    Returns:
        Array of expected payoffs, one per action.
    """
    num_players = len(strategies)
    # Contract over all dimensions except the player's
    result = payoff_tensor.copy()
    # Sum from last dimension backwards, skipping player
    dims_to_contract: list[int] = []
    for i in range(num_players - 1, -1, -1):
        if i != player:
            dims_to_contract.append(i)

    for dim in sorted(dims_to_contract, reverse=True):
        result = np.tensordot(result, strategies[dim], axes=([dim], [0]))

    return result


def _solve_support(
    payoff_1: np.ndarray,
    payoff_2: np.ndarray,
    support_1: list[int],
    support_2: list[int],
    tol: float,
) -> NashEquilibrium | None:
    """Try to find a NE with given supports.

    For supports S1, S2, we need:
    - Player 2's strategy y makes player 1 indifferent on S1
    - Player 1's strategy x makes player 2 indifferent on S2
    - Both x, y are valid probability distributions
    - No action outside support gives higher payoff

    Returns None if no valid equilibrium exists for these
    supports.
    """
    s1_arr = np.array(support_1)
    s2_arr = np.array(support_2)

    # Extract sub-matrices
    a_sub = payoff_1[np.ix_(s1_arr, s2_arr)]
    b_sub = payoff_2[np.ix_(s1_arr, s2_arr)]

    # Solve for y (makes player 1 indifferent)
    y = _solve_indifference(a_sub, tol)
    if y is None:
        return None

    # Solve for x (makes player 2 indifferent)
    x = _solve_indifference(b_sub.T, tol)
    if x is None:
        return None

    # Build full strategy vectors
    m, n = payoff_1.shape
    full_x = np.zeros(m)
    full_y = np.zeros(n)
    full_x[s1_arr] = x
    full_y[s2_arr] = y

    # Vectorized best response check
    # All payoffs for player 1 when opponent plays y
    payoffs_1 = payoff_1 @ full_y
    v1 = float(full_x @ payoffs_1)
    # Check no outside action beats the equilibrium payoff
    outside_1 = np.ones(m, dtype=bool)
    outside_1[s1_arr] = False
    if outside_1.any() and np.any(payoffs_1[outside_1] > v1 + tol):
        return None

    # All payoffs for player 2 when opponent plays x
    payoffs_2 = payoff_2.T @ full_x
    v2 = float(full_y @ payoffs_2)
    outside_2 = np.ones(n, dtype=bool)
    outside_2[s2_arr] = False
    if outside_2.any() and np.any(payoffs_2[outside_2] > v2 + tol):
        return None

    return NashEquilibrium(
        strategies={"player_0": full_x, "player_1": full_y},
        payoffs={"player_0": v1, "player_1": v2},
        support={
            "player_0": support_1,
            "player_1": support_2,
        },
    )


def _solve_indifference(
    matrix: np.ndarray,
    tol: float,
) -> np.ndarray | None:
    """Solve for a mixed strategy that makes the opponent indifferent.

    Given matrix M (k x k), find probability vector p (length k)
    such that M @ p = v * 1 and sum(p) = 1, p >= 0.

    For square systems (k == cols), uses direct solve for speed.
    """
    k, cols = matrix.shape

    # Build the linear system: differences + sum = 1
    a = np.empty((k, cols))
    a[:-1] = matrix[1:] - matrix[0]
    a[-1] = 1.0
    b_vec = np.zeros(k)
    b_vec[-1] = 1.0

    try:
        if k == cols:
            result = np.linalg.solve(a, b_vec)
        else:
            result, _, _, _ = np.linalg.lstsq(
                a,
                b_vec,
                rcond=None,
            )
            err = np.max(np.abs(a @ result - b_vec))
            if err > tol * 100:
                return None
    except np.linalg.LinAlgError:
        return None

    # Fast non-negativity check (most common rejection)
    if result.min() < -tol:
        return None

    result = np.maximum(result, 0.0)
    result_sum = result.sum()
    if abs(result_sum - 1.0) > tol * 100:
        return None

    result /= result_sum
    return result


def _deduplicate(
    equilibria: list[NashEquilibrium],
    tol: float,
) -> list[NashEquilibrium]:
    """Remove duplicate equilibria based on strategy similarity."""
    if not equilibria:
        return equilibria

    unique: list[NashEquilibrium] = [equilibria[0]]
    for eq in equilibria[1:]:
        is_dup = False
        for existing in unique:
            all_close = all(
                np.allclose(
                    eq.strategies[pid],
                    existing.strategies[pid],
                    atol=tol * 100,
                )
                for pid in eq.strategies
            )
            if all_close:
                is_dup = True
                break
        if not is_dup:
            unique.append(eq)
    return unique
