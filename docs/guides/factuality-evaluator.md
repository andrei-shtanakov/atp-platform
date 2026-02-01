# Factuality Evaluator Guide

The Factuality Evaluator verifies the factual accuracy of agent outputs against trusted sources. It extracts factual claims, verifies them against ground truth data, detects potential hallucinations, and validates citations.

## Overview

The evaluator provides:

- **Claim Extraction**: Automatically extracts dates, numbers, quotes, statistics, and factual statements
- **Ground Truth Verification**: Verifies claims against a JSON file of known facts
- **LLM-Based Verification**: Uses Claude to verify claims when ground truth is insufficient
- **Citation Extraction**: Extracts and validates URL and reference citations
- **Hallucination Detection**: Identifies potential hallucination indicators like vague language, overconfidence, and inconsistencies
- **Confidence Scoring**: Assigns confidence scores to each claim

## Quick Start

### Basic Usage

```yaml
assertions:
  - type: "factuality"
    config:
      ground_truth_file: "facts.json"
      verification_method: "ground_truth"
      min_confidence: 0.8
```

### With LLM Verification

```yaml
assertions:
  - type: "factuality"
    config:
      ground_truth_file: "facts.json"
      verification_method: "llm_verify"
      min_confidence: 0.8
      check_citations: true
      detect_hallucinations: true
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ground_truth_file` | string | - | Path to JSON file with ground truth facts |
| `verification_method` | string | "ground_truth" | Verification method: "ground_truth", "llm_verify", or "rag" |
| `min_confidence` | float | 0.8 | Minimum confidence threshold for passing |
| `check_citations` | bool | true | Whether to extract and validate citations |
| `detect_hallucinations` | bool | true | Whether to run hallucination detection |
| `path` | string | - | Optional artifact path to evaluate |

## Ground Truth File Format

Create a JSON file with the following structure:

```json
{
  "facts": {
    "topic_name": "expected_value",
    "AI founding year": "1956",
    "founder_name": "John McCarthy"
  },
  "dates": {
    "AI founded": "1956",
    "OpenAI founded": "2015"
  },
  "numbers": {
    "enterprise_adoption_rate": "75",
    "initial_investment": "1 billion"
  },
  "valid_sources": [
    "https://trusted-source.com",
    "McCarthy, J. (1956)"
  ]
}
```

### Sections

- **facts**: Key-value pairs of general facts
- **dates**: Date-related facts with key descriptions
- **numbers**: Numeric facts (percentages, amounts, etc.)
- **valid_sources**: List of trusted citation sources

## Verification Methods

### Ground Truth (`ground_truth`)

Verifies claims only against the provided ground truth file. Fast and deterministic but limited to known facts.

```yaml
verification_method: "ground_truth"
```

### LLM Verify (`llm_verify`)

First checks ground truth, then uses Claude to verify remaining claims. More comprehensive but incurs API costs.

```yaml
verification_method: "llm_verify"
```

### RAG (`rag`)

Combines ground truth verification with LLM verification, treating ground truth as a knowledge base.

```yaml
verification_method: "rag"
```

## Claim Types

The evaluator extracts these claim types:

| Type | Description | Example |
|------|-------------|---------|
| `date` | Date-related claims | "founded in 1956" |
| `number` | Numeric claims | "$1 billion investment" |
| `statistic` | Percentage/statistical claims | "75% of enterprises" |
| `quote` | Quoted text | "He said 'AI will change everything'" |
| `fact` | General factual statements | "AI was founded by John McCarthy" |

## Hallucination Detection

The evaluator detects these hallucination indicators:

### Vague Language
- "many experts believe..."
- "it is well-known that..."
- "some say..."

### Overconfident Language
- "definitely", "certainly", "undoubtedly"
- "always", "never"
- "everyone", "no one"

### Unverifiable Specifics
- Precise percentages without sources
- Exact numbers without citations

### Inconsistencies
- Conflicting dates or numbers within the same document

## Scoring

The overall score is calculated as:

```
score = (claim_score × 0.6) + (citation_score × 0.2) + ((1 - hallucination_penalty) × 0.2)
```

Where:
- **claim_score**: Ratio of verified claims, with penalty for false claims
- **citation_score**: Ratio of valid citations
- **hallucination_penalty**: Accumulated penalty from hallucination indicators

## Example Test Suite

```yaml
test_suite: "Factuality Tests"
version: "1.0"

tests:
  - id: "fact-001"
    name: "Verify AI History Report"
    task:
      description: "Write a factual report about AI history"
    assertions:
      - type: "factuality"
        config:
          ground_truth_file: "ai_facts.json"
          verification_method: "llm_verify"
          min_confidence: 0.8
          check_citations: true
          detect_hallucinations: true

  - id: "fact-002"
    name: "Verify Company Data"
    task:
      description: "Summarize company financial data"
    assertions:
      - type: "factuality"
        config:
          ground_truth_file: "company_data.json"
          verification_method: "ground_truth"
          min_confidence: 0.9
```

## Evaluation Results

The evaluation result includes detailed information:

```json
{
  "name": "factuality_check",
  "passed": true,
  "score": 0.85,
  "message": "8/10 claims verified; 1 hallucination indicators",
  "details": {
    "total_claims": 10,
    "verified_claims": 8,
    "unverified_claims": 1,
    "false_claims": 1,
    "total_citations": 3,
    "valid_citations": 2,
    "hallucination_indicators": 1,
    "high_severity_indicators": 0,
    "min_confidence_threshold": 0.8,
    "claims": [...],
    "citations": [...],
    "hallucinations": [...],
    "llm_tokens": {
      "input": 1500,
      "output": 300
    }
  }
}
```

## Python API

### Using the Evaluator Directly

```python
from atp.evaluators import FactualityEvaluator, FactualityConfig

# Create evaluator with custom config
config = FactualityConfig(
    api_key="your-api-key",
    model="claude-sonnet-4-20250514",
    temperature=0.0,
)
evaluator = FactualityEvaluator(config)

# Evaluate
result = await evaluator.evaluate(task, response, trace, assertion)
```

### Using Individual Components

```python
from atp.evaluators.factuality import (
    ClaimExtractor,
    CitationExtractor,
    GroundTruthVerifier,
    HallucinationDetector,
)

# Extract claims
extractor = ClaimExtractor()
claims = extractor.extract_claims(text)

# Verify against ground truth
verifier = GroundTruthVerifier("facts.json")
verified_claims = verifier.verify_claims(claims)

# Extract citations
citation_extractor = CitationExtractor()
citations = citation_extractor.extract_citations(text)

# Detect hallucinations
detector = HallucinationDetector()
indicators = detector.detect(text, claims)
```

## Best Practices

1. **Comprehensive Ground Truth**: Include as many verifiable facts as possible in your ground truth file

2. **Use LLM Verification Sparingly**: The `llm_verify` method incurs API costs; use `ground_truth` for deterministic tests

3. **Set Appropriate Thresholds**: Adjust `min_confidence` based on your use case:
   - 0.9+ for critical factual accuracy
   - 0.7-0.8 for general content
   - 0.5-0.6 for exploratory tests

4. **Monitor Hallucination Indicators**: High-severity indicators (inconsistencies) are more concerning than low-severity (vague language)

5. **Validate Citations**: Enable `check_citations` for content that should reference sources

## Troubleshooting

### No Claims Extracted

- Check if the content contains factual statements
- Verify the artifact path is correct
- Ensure content is not empty

### Low Confidence Scores

- Add more facts to ground truth file
- Use `llm_verify` for better coverage
- Check if claims are verifiable

### High Hallucination Count

- Review content for unsourced claims
- Add citations for specific statistics
- Remove overconfident language

## See Also

- [LLM Judge Evaluator](../05-evaluators.md#llm-judge)
- [Evaluator Configuration](../reference/configuration.md)
- [Test Format Reference](../reference/test-format.md)
