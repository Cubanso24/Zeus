# Training Data Format

## Overview
This directory contains training data for fine-tuning the Splunk Query LLM. The data should be in JSONL (JSON Lines) format, with each line representing a training example.

## Data Format

### Standard Query Format
Each training example should follow this structure:

```json
{
  "instruction": "User's natural language request",
  "input": "Additional context or constraints (optional)",
  "output": "SPL (Splunk Processing Language) query"
}
```

### Clarification Request Format
For examples where the LLM should ask for clarification:

```json
{
  "instruction": "Ambiguous or incomplete user request",
  "input": "",
  "output": "CLARIFICATION: What specific information do you need? [LLM asks for missing details]"
}
```

## Example Categories

### 1. Basic Search Queries
Simple searches, filtering, and field extraction

### 2. Aggregation and Statistics
Queries involving stats, timechart, chart commands

### 3. Security Use Cases
- Failed login attempts
- Malware detection
- Network anomalies
- User behavior analytics
- Threat hunting

### 4. Performance Monitoring
- System metrics
- Application performance
- Resource utilization

### 5. Complex Multi-Stage Queries
- Subsearches
- Joins
- Lookups
- Transaction analysis

## Data Quality Guidelines

1. **Accuracy**: All SPL queries must be syntactically correct and executable
2. **Diversity**: Include various query patterns, commands, and use cases
3. **Realism**: Based on actual cybersecurity analyst workflows
4. **Clarity**: Instructions should represent how analysts naturally describe their needs
5. **Completeness**: Include both simple and complex scenarios

## File Naming Convention

- `train_basic.jsonl` - Basic search and filtering
- `train_security.jsonl` - Security-specific queries
- `train_aggregation.jsonl` - Statistical analysis queries
- `train_advanced.jsonl` - Complex multi-stage queries
- `train_clarification.jsonl` - Examples requiring clarification

## Adding New Data

When adding new training examples:
1. Ensure queries are tested in Splunk
2. Validate JSON format
3. Include diverse phrasings for similar queries
4. Document any assumptions or prerequisites
5. Use realistic index names, sourcetypes, and field names
