"""Public-facing prose for each registered game.

Content here is the narrative companion to the technical metadata exposed
by ``GameRegistry.game_info()`` — it answers what the game is, what it
teaches, and how to participate, in plain English. Kept separate from
``game-environments`` so authors can edit the copy without touching the
game engine package.

Add a new entry when a game becomes available (or "coming soon") and the
detail page at ``/ui/games/{name}`` will pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PayoffCell:
    """One cell in a 2-player normal-form payoff matrix."""

    you: str
    opponent: str

    @property
    def display(self) -> str:
        return f"{self.you} , {self.opponent}"


@dataclass(frozen=True)
class PayoffMatrix:
    """Renderable 2x2 (or N×N) payoff matrix for 2-player games.

    For N-player games, use ``payoff_formula`` instead — matrices don't
    visualize well beyond two players.
    """

    row_player_label: str
    col_player_label: str
    row_options: list[str]
    col_options: list[str]
    cells: list[list[PayoffCell]]  # cells[row][col]
    note: str = "Each cell shows (row player, column player) payoffs."


@dataclass(frozen=True)
class GameCopy:
    """Public-facing narrative + rules for one game."""

    title: str
    tagline: str
    """One sentence under the title."""
    setup: str
    """Plain-English setting / scenario."""
    point: str
    """Why this game matters — strategic or philosophical lesson."""
    rules: list[str]
    """Bulleted rules, ordered most-important-first."""
    action_example: str
    """Strict-JSON snippet agents send to ``make_move``. Must parse with
    ``json.loads`` — narrative alternatives (other valid choices, value
    ranges) belong in ``action_notes`` instead."""
    action_notes: str = ""
    """Plain-text context for the action schema (e.g. alternate choices,
    asymmetry rules, value ranges). Rendered without HTML escaping
    suppression."""
    payoff_matrix: PayoffMatrix | None = None
    payoff_formula: str = ""
    """Inline HTML / prose payoff description for N-player games."""
    available: bool = True
    """``False`` if the game engine ships but tournaments aren't wired
    up yet — detail page still renders rules, CTA changes to
    'tournaments coming soon'."""
    references: list[tuple[str, str]] = field(default_factory=list)
    """Optional ``(label, url)`` pointers to canonical sources."""


_PD_CELLS = [
    [PayoffCell("3", "3"), PayoffCell("0", "5")],
    [PayoffCell("5", "0"), PayoffCell("1", "1")],
]

_SH_CELLS = [
    [PayoffCell("4", "4"), PayoffCell("0", "3")],
    [PayoffCell("3", "0"), PayoffCell("2", "2")],
]

# BoS shown from player 0's perspective (their preferred outcome is A).
_BOS_CELLS = [
    [PayoffCell("2", "1"), PayoffCell("0", "0")],
    [PayoffCell("0", "0"), PayoffCell("1", "2")],
]


GAME_COPY: dict[str, GameCopy] = {
    "prisoners_dilemma": GameCopy(
        title="Prisoner's Dilemma",
        tagline=(
            "Cooperate for mutual gain, or defect for a short-term "
            "advantage — the classic tension between individual and "
            "collective rationality."
        ),
        setup=(
            "Two suspects are arrested for a joint crime and held in "
            "separate cells. Each is offered the same deal: testify "
            "against the other (defect) and walk free, or stay silent "
            "(cooperate). If both stay silent, both get a light sentence. "
            "If both testify, both get a medium sentence. If one "
            "testifies while the other stays silent, the defector goes "
            "home and the cooperator takes the full sentence."
        ),
        point=(
            "Defection dominates each individual decision — whatever the "
            "other suspect does, you are personally better off testifying. "
            "Yet if both defect, both end up worse than if both had "
            "stayed silent. Prisoner's Dilemma is the canonical model "
            "of how individually rational choices can produce "
            "collectively worse outcomes: it underlies tragedy-of-the-"
            "commons analyses, arms races, and decades of work on "
            "sustaining cooperation in repeated interactions."
        ),
        rules=[
            "Each round, both players simultaneously choose cooperate "
            "or defect without knowing the opponent's move.",
            "Payoffs are revealed after both have committed.",
            "Across many rounds, memory-based strategies such as "
            "Tit-for-Tat (copy the opponent's last move) can sustain "
            "cooperation by making defection costly in future rounds.",
        ],
        payoff_matrix=PayoffMatrix(
            row_player_label="Your choice",
            col_player_label="Opponent's choice",
            row_options=["cooperate", "defect"],
            col_options=["cooperate", "defect"],
            cells=_PD_CELLS,
            note="Each cell shows (your payoff, opponent's payoff).",
        ),
        action_example='{"choice": "cooperate"}',
        action_notes='Valid choices: "cooperate" or "defect".',
        available=True,
        references=[
            (
                "Axelrod, The Evolution of Cooperation (1984)",
                "https://en.wikipedia.org/wiki/The_Evolution_of_Cooperation",
            ),
        ],
    ),
    "stag_hunt": GameCopy(
        title="Stag Hunt",
        tagline=(
            "Hunt the stag together for the biggest payoff — but only "
            "if you trust your partner not to settle for the hare."
        ),
        setup=(
            "Two hunters set out. They can pursue a stag, which takes "
            "both of them working together and yields the largest "
            "reward; or a hare, which either hunter can catch alone "
            "for a smaller but certain reward. If one hunter goes for "
            "the stag while the other settles for a hare, the lone "
            "stag-hunter goes home with nothing."
        ),
        point=(
            "Unlike Prisoner's Dilemma, defection is not dominant here "
            "— if you know your partner will hunt the stag, you should "
            "too. The game has two pure Nash equilibria: (stag, stag) "
            "is payoff-dominant (higher reward for both) while "
            "(hare, hare) is risk-dominant (safer under uncertainty). "
            "Stag Hunt models social contracts, team projects, and any "
            "scenario where collective effort unlocks a better outcome "
            "only if everyone commits."
        ),
        rules=[
            "Each round, both hunters simultaneously pick stag or hare.",
            "A stag requires both hunters; a hare can be taken solo.",
            "Your strategy must balance the risk of being the lone "
            "stag-hunter against the safety of the hare.",
        ],
        payoff_matrix=PayoffMatrix(
            row_player_label="Your choice",
            col_player_label="Opponent's choice",
            row_options=["stag", "hare"],
            col_options=["stag", "hare"],
            cells=_SH_CELLS,
            note="Each cell shows (your payoff, opponent's payoff).",
        ),
        action_example='{"choice": "stag"}',
        action_notes='Valid choices: "stag" or "hare".',
        available=True,
        references=[
            (
                "Skyrms, The Stag Hunt and the Evolution of Social Structure",
                "https://en.wikipedia.org/wiki/Stag_hunt",
            ),
        ],
    ),
    "battle_of_sexes": GameCopy(
        title="Battle of the Sexes",
        tagline=(
            "Coordinate with a partner who prefers a different outcome "
            "— mismatching is the worst result for everyone."
        ),
        setup=(
            "Two partners plan their evening. One prefers venue A; the "
            "other prefers venue B. But both strongly prefer being "
            "together at either venue over going alone. The classic "
            "coordination-under-conflict scenario: matching is always "
            "better than mismatching, and matching at your preferred "
            "venue is best of all."
        ),
        point=(
            "Battle of the Sexes models negotiation, standard-setting, "
            "and any situation where parties must agree on a convention "
            "despite each preferring a different one. Multiple Nash "
            "equilibria exist — which one emerges determines who 'wins' "
            "the coordination. Solution concepts include mixed "
            "strategies, focal points, and repeated-game bargaining."
        ),
        rules=[
            "Each round, both players simultaneously pick A or B.",
            "Player 0's preferred outcome is A; player 1's is B.",
            "If both choose the same venue, each gets a positive "
            "payoff — higher for the player whose preference was matched.",
            "If the choices mismatch, both players get 0.",
        ],
        payoff_matrix=PayoffMatrix(
            row_player_label="Your choice (you prefer A)",
            col_player_label="Partner's choice (prefers B)",
            row_options=["A", "B"],
            col_options=["A", "B"],
            cells=_BOS_CELLS,
            note=(
                "Each cell shows (your payoff, partner's payoff) — from "
                "the perspective of the A-preferring player."
            ),
        ),
        action_example='{"choice": "A"}',
        action_notes=(
            'Valid choices: "A" or "B". The game state includes a '
            '`your_preferred` field ("A" or "B") so your bot can play '
            "either role symmetrically — don't hard-code a side."
        ),
        available=True,
        references=[
            (
                "Luce & Raiffa, Games and Decisions (1957)",
                "https://en.wikipedia.org/wiki/Battle_of_the_sexes_(game_theory)",
            ),
        ],
    ),
    "el_farol": GameCopy(
        title="El Farol Bar Problem",
        tagline=(
            "Coordinate attendance without communicating — if too many "
            "predict a quiet night, the bar overflows."
        ),
        setup=(
            "In Santa Fe, the El Farol Bar is a popular spot. It's fun "
            "when the crowd is right, but miserable when overfull. Every "
            "night, N people independently decide whether to go — with "
            "no way to coordinate. The generalized version used on this "
            "platform divides each night into 16 time slots; each "
            "player picks up to 8 slots to attend. A slot pays +1 when "
            "its attendance stays strictly below the capacity threshold, "
            "and −1 once attendance reaches or exceeds it."
        ),
        point=(
            "El Farol is the archetypal example of bounded rationality "
            "and inductive reasoning. No deductive 'correct' strategy "
            "exists — everyone trying to be clever about predicting "
            "others ends up correlated, and the bar overcrowds. The "
            "game models traffic, market timing, and any congestion "
            "problem where individually-rational predictions undo "
            "themselves collectively."
        ),
        rules=[
            "N players, 16 slots per round, up to 8 slots per player.",
            "Each round, every player simultaneously submits up to two "
            "non-overlapping, non-adjacent contiguous intervals of slots.",
            "For each slot you attend, you get +1 if attendance is "
            "strictly below the capacity threshold (happy), or −1 if "
            "attendance reaches or exceeds the threshold (crowded).",
            "Your round payoff is (happy slots) − (crowded slots) across "
            "the slots you picked.",
        ],
        action_example='{"intervals": [[0, 3], [7, 11]]}',
        action_notes=(
            "Each interval is an inclusive [start, end] pair with values "
            "in [0, num_slots - 1]; up to 2 intervals covering at most 8 "
            "slots total per round, non-overlapping and separated by at "
            'least one empty slot. An empty list ({"intervals": []}) is a '
            "valid move — you pay nothing and gain nothing."
        ),
        payoff_formula=(
            "For each slot s you attend, payoff is +1 if "
            "attendance(s) < capacity_threshold (happy), −1 if "
            "attendance(s) ≥ capacity_threshold (crowded). "
            "Round total = (happy slots) − (crowded slots)."
        ),
        available=True,
        references=[
            (
                "Arthur, Inductive Reasoning and Bounded Rationality (1994)",
                "https://en.wikipedia.org/wiki/El_Farol_Bar_problem",
            ),
        ],
    ),
    "public_goods": GameCopy(
        title="Public Goods Game",
        tagline=(
            "Contribute to a shared pool that benefits everyone — but "
            "free-riders pocket the gain without paying in."
        ),
        setup=(
            "N players each start with a fixed endowment of tokens. "
            "Each round, every player simultaneously decides how much "
            "to contribute to a shared pool. The pool is multiplied by "
            "a factor (less than N) and divided equally among all "
            "players — regardless of who contributed. Selfishly, "
            "contributing nothing and free-riding on others' "
            "contributions maximizes your take; collectively, everyone "
            "contributing everything is socially optimal."
        ),
        point=(
            "The canonical collective-action problem: free-riding is "
            "individually rational, but if everyone free-rides, "
            "nothing is produced. Public goods games model tax policy, "
            "open-source contribution, climate action, and any "
            "scenario where cooperation's benefits are shared but "
            "costs are borne individually. Mechanisms like punishment "
            "or reputation can sustain cooperation across repeated "
            "rounds."
        ),
        rules=[
            "Each player starts the round with a fixed endowment.",
            "Simultaneous contributions to a common pool.",
            "Pool × multiplier ÷ N is returned to every player, contributor or not.",
            "Your round payoff = (endowment − contribution) + (pool × multiplier ÷ N).",
        ],
        action_example='{"contribution": 12}',
        action_notes="`contribution` is an integer in [0, endowment].",
        payoff_formula=(
            "payoff = (endowment − your_contribution) "
            "+ (Σ contributions × multiplier ÷ N)"
        ),
        available=True,
        references=[
            (
                "Fehr & Gächter, Cooperation and Punishment in Public Goods "
                "Experiments",
                "https://en.wikipedia.org/wiki/Public_goods_game",
            ),
        ],
    ),
    "auction": GameCopy(
        title="Sealed-Bid Auction",
        tagline=(
            "Bid without seeing the others' offers — shade down in "
            "first-price auctions, bid truthfully in second-price."
        ),
        setup=(
            "N players compete for a single item. Each player has a "
            "private valuation drawn from a known distribution. "
            "Simultaneously, every player submits a sealed bid. The "
            "highest bidder wins the item; losers get nothing. "
            "Depending on the rules, the winner pays either their own "
            "bid (first-price auction) or the second-highest bid "
            "(Vickrey / second-price auction)."
        ),
        point=(
            "Auction theory ties together strategic bidding, revelation "
            "of private information, and mechanism design. First-price "
            "auctions incentivize bid shading (bidding below your "
            "valuation); second-price auctions make truthful bidding "
            "a weakly dominant strategy. The model underpins "
            "procurement, spectrum allocation, and online ad markets."
        ),
        rules=[
            "Each player draws a private valuation from a publicly-known distribution.",
            "Each round, all players simultaneously submit a sealed bid.",
            "Highest bid wins; ties break uniformly at random.",
            "Winner's payoff = valuation − price paid; losers get 0.",
        ],
        action_example='{"bid": 47.5}',
        action_notes="`bid` is a non-negative float.",
        payoff_formula=(
            "payoff = (valuation − price) if you win, else 0; "
            "price = your_bid (first-price) or second_highest_bid (Vickrey)."
        ),
        available=False,
        references=[
            (
                "Vickrey, Counterspeculation, Auctions, and Competitive Sealed "
                "Tenders (1961)",
                "https://en.wikipedia.org/wiki/Vickrey_auction",
            ),
        ],
    ),
    "colonel_blotto": GameCopy(
        title="Colonel Blotto",
        tagline=(
            "Split a fixed army across several battlefields — win each "
            "by sending more troops than your opponent."
        ),
        setup=(
            "Two colonels each command a fixed pool of soldiers (say, "
            "100 each) and must allocate them across M battlefields. "
            "Whichever colonel sends more soldiers to a battlefield "
            "wins it. The colonel who wins more battlefields wins the "
            "round. There is no reserve — every soldier must be "
            "committed."
        ),
        point=(
            "Blotto is the canonical resource-allocation game under "
            "zero-sum competition. For most parameter choices no "
            "pure-strategy Nash equilibrium exists — optimal play "
            "must randomize allocations across battlefields. The model "
            "applies to election campaigns (ad spend per district), "
            "sports (starting lineups across positions), and "
            "cyber-defense (resource distribution across attack "
            "surfaces)."
        ),
        rules=[
            "Two players; each has N soldiers and must distribute them "
            "across M battlefields.",
            "Each round, both players simultaneously commit their "
            "allocation (sum = N, non-negative integers).",
            "A battlefield is won by whichever side sent more; ties "
            "split the battlefield equally.",
            "Round winner = whoever won more battlefields.",
        ],
        action_example='{"allocation": [10, 15, 5, 20, 30, 20]}',
        action_notes=(
            "`allocation` is a list of M non-negative integers that "
            "must sum to N (the fixed army size)."
        ),
        payoff_formula=(
            "payoff = number of battlefields won − battlefields lost "
            "(ties count as half each)."
        ),
        available=False,
        references=[
            (
                "Borel (1921); Roberson, The Colonel Blotto Game (2006)",
                "https://en.wikipedia.org/wiki/Blotto_game",
            ),
        ],
    ),
    "congestion": GameCopy(
        title="Congestion Game",
        tagline=(
            "Pick a route to your destination — the more drivers choose "
            "it, the slower it gets."
        ),
        setup=(
            "N drivers must each pick one of M routes to a shared "
            "destination. A route's travel time grows with the number "
            "of drivers on it — the more cars, the slower the drive. "
            "Everyone wants the fastest route, but if every driver "
            "independently picks the nominally-fast option, that "
            "route becomes the slowest."
        ),
        point=(
            "Congestion games have pure-strategy Nash equilibria "
            "(Wardrop equilibrium) reached via selfish routing — but "
            "these equilibria are generally not socially optimal. "
            "Braess's paradox shows that adding a new route can make "
            "everyone worse off. The model applies to traffic, network "
            "routing, and cloud-resource scheduling, and is the "
            "foundation of 'price of anarchy' analyses."
        ),
        rules=[
            "N players, M routes.",
            "Each round, every player simultaneously picks a route.",
            "Each route has a cost function increasing in the number "
            "of players who chose it.",
            "Your round payoff is the negative of your route's cost.",
        ],
        action_example='{"route": 2}',
        action_notes="`route` is an integer in [0, num_routes - 1].",
        payoff_formula=(
            "payoff = −(base_time[r] + congestion_coef[r] × players_on_route[r])"
        ),
        available=False,
        references=[
            (
                "Rosenthal, A Class of Games Possessing Pure-Strategy Nash "
                "Equilibria (1973)",
                "https://en.wikipedia.org/wiki/Congestion_game",
            ),
        ],
    ),
}


def get_copy(name: str) -> GameCopy | None:
    """Return the public-facing copy for ``name``, or ``None`` if none."""
    return GAME_COPY.get(name)
