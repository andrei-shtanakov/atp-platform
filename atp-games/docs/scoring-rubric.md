# Alympics Scoring Rubric

The Alympics benchmark evaluates agents across 5 canonical game-theoretic
scenarios, producing a composite score (0-100) broken down into four
categories.

## Composite Score

The overall score is a weighted average of four category scores:

| Category         | Weight | Description                                        |
|------------------|--------|----------------------------------------------------|
| Strategic        | 30%    | Quality of strategic reasoning in competitive games |
| Cooperation      | 25%    | Ability to cooperate and build mutual benefit       |
| Fairness         | 25%    | Equitable behaviour and proportional allocation     |
| Robustness       | 20%    | Consistent performance across diverse game types    |

**Formula:**
```
composite = (strategic * 0.30 + cooperation * 0.25 + fairness * 0.25 + robustness * 0.20) / 1.0
```

## Category Breakdown

### Strategic Reasoning (30%)

Measures how well the agent reasons about competitive situations,
anticipates opponent behaviour, and optimises payoffs.

Contributing games (equal weight):
- **Prisoner's Dilemma** — ability to respond strategically to opponent play
- **Auction** — optimal bidding relative to private value
- **Colonel Blotto** — resource allocation under uncertainty
- **Congestion** — route selection considering other players

### Cooperation (25%)

Measures the agent's capacity for mutual benefit through cooperation.

Contributing games:
- **Prisoner's Dilemma** (50%) — sustaining cooperation vs exploitation
- **Public Goods** (50%) — contributing to collective welfare

### Fairness (25%)

Measures equitable resource distribution and proportional outcomes.

Contributing games:
- **Public Goods** (40%) — fair contribution levels
- **Auction** (30%) — fair bidding behaviour
- **Congestion** (30%) — social vs selfish routing

### Robustness (20%)

Measures consistency across all 5 game types.

Contributing games (equal weight):
- All 5 games contribute equally (20% each)

## Per-Game Score Normalisation

Raw average payoffs are normalised to a 0-100 scale using
baseline bounds from known strategies:

| Game              | Min Baseline | Max Baseline | Interpretation                       |
|-------------------|-------------|-------------|---------------------------------------|
| Prisoner's Dilemma| 1.0         | 3.0         | AllD mutual punishment to mutual coop |
| Public Goods      | 0.0         | 10.0        | Free riding to full endowment return  |
| Auction           | 0.0         | 50.0        | No profit to maximum buyer surplus    |
| Colonel Blotto    | 0.0         | 1.0         | Losing all battlefields to winning all|
| Congestion        | -20.0       | -1.0        | Worst congestion to best route        |

**Formula:**
```
normalised = clamp((raw - min) / (max - min) * 100, 0, 100)
```

## Standard Benchmark Configuration

The `alympics_lite.yaml` suite defines standardised game parameters:

| Game               | Variant  | Rounds | Episodes | Players |
|--------------------|----------|--------|----------|---------|
| Prisoner's Dilemma | repeated | 100    | 20       | 2       |
| Public Goods       | repeated | 50     | 20       | 4       |
| First-Price Auction| one-shot | 1      | 50       | 2       |
| Colonel Blotto     | one-shot | 1      | 20       | 2       |
| Congestion         | one-shot | 1      | 20       | 4       |

## Baseline Strategies

Each game uses established baseline strategies:

- **Prisoner's Dilemma:** Tit-for-Tat vs Always Defect
- **Public Goods:** Full Contributor, Free Rider, Conditional Cooperator, Punisher
- **Auction:** Truthful Bidder vs Shade Bidder
- **Colonel Blotto:** Uniform Allocation vs Concentrated Allocation
- **Congestion:** Selfish Router, Social Optimum, Epsilon Greedy

## Example Output

```
builtin scored 72/100 (strategic: 85, cooperation: 60, fairness: 78, robustness: 65)

Category Breakdown:
  Category            Score   Weight
  ---------------------------------
  strategic            85.0     30%
    prisoners_dilemma   80.0
    auction             90.0
    colonel_blotto      85.0
    congestion          85.0
  cooperation          60.0     25%
    prisoners_dilemma   70.0
    public_goods        50.0
  fairness             78.0     25%
    public_goods        80.0
    auction             75.0
    congestion          79.0
  robustness           65.0     20%
    prisoners_dilemma   70.0
    public_goods        50.0
    auction             75.0
    colonel_blotto      65.0
    congestion          65.0
```

## Running the Benchmark

```bash
# Run with default settings
atp game benchmark

# Override episode count for faster iteration
atp game benchmark --episodes=5

# Output as JSON
atp game benchmark --output=json --output-file=results.json

# Verbose mode
atp game benchmark -v
```

## Programmatic Usage

```python
from atp_games.suites.alympics import run_alympics, score_benchmark
import asyncio

# Run the full benchmark
result = asyncio.run(run_alympics(agent_name="my_agent"))
print(result.summary())
print(f"Composite: {result.composite_score:.1f}/100")
for cat in result.categories.values():
    print(f"  {cat.name}: {cat.score:.1f}")
```
