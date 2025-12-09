#!/usr/bin/env python3
"""
Generate validated training queries from 13 months of Wazuh logs.

This script:
1. Connects to Splunk to understand available data
2. Generates natural language instructions and corresponding queries
3. Validates each query by running it against Splunk
4. Saves successful queries to the approved training dataset
"""

import os
import sys
import json
import time
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
requests.packages.urllib3.disable_warnings()

# Configuration
SPLUNK_HOST = os.getenv("SPLUNK_HOST", "https://10.10.10.114:8089")
SPLUNK_TOKEN = os.getenv("SPLUNK_TOKEN", "eyJraWQiOiJzcGx1bmsuc2VjcmV0IiwiYWxnIjoiSFM1MTIiLCJ2ZXIiOiJ2MiIsInR0eXAiOiJzdGF0aWMifQ.eyJpc3MiOiJhZmVybmFuZGV6IGZyb20gc3BzZWFyY2giLCJzdWIiOiJhZmVybmFuZGV6IiwiYXVkIjoiSW1wcm92aW5nIFpldXMgdG9vbCIsImlkcCI6IkxEQVA6Ly9PcGVuTERBUCIsImp0aSI6IjliODJmYWU1OTg2Y2YyZjhkZDczMWY4MWVlYTg5YWRkMjE0MGQ2MGQ2OGY3YTJhZTNjMjMxMmQ2MGEwNzQ2OTUiLCJpYXQiOjE3NjUyMjA4MzgsImV4cCI6MTc2NzgxMjgzOCwibmJyIjoxNzY1MjIwODM4fQ.sxGtJMsayNdKNW8wj9zeT2Yf0Y9yMmulLgsPL3QrulzvAsfchS0qgZTnAnMcaUa5sirQowQp8fFhfAkLah07bg")
OUTPUT_FILE = "data/feedback/generated_training.jsonl"


@dataclass
class QueryTemplate:
    """Template for generating training data."""
    category: str
    instruction_templates: List[str]
    query_template: str
    variations: Dict[str, List[str]]


# Query templates based on real Wazuh data patterns
QUERY_TEMPLATES = [
    # Authentication queries
    QueryTemplate(
        category="authentication",
        instruction_templates=[
            "Find failed authentication attempts {time_range}",
            "Show me failed login attempts {time_range}",
            "Get all authentication failures {time_range}",
            "List failed SSH login attempts {time_range}",
            "Find brute force login attempts {time_range}",
            "Show authentication failures by agent {time_range}",
            "Count failed logins per user {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} rule.groups{{}}="authentication_failed" | {aggregation}',
        variations={
            "time_range": ["in the last hour", "in the last 24 hours", "in the last 7 days", "in the last 30 days", "from the last week"],
            "earliest": ["-1h", "-24h", "-7d", "-30d", "-7d"],
            "aggregation": [
                "stats count by agent.name, rule.description",
                "stats count by agent.name",
                "table _time agent.name rule.description rule.level",
                "stats count by agent.name | sort -count",
            ]
        }
    ),
    QueryTemplate(
        category="authentication",
        instruction_templates=[
            "Find successful authentication events {time_range}",
            "Show me successful logins {time_range}",
            "Get successful SSH sessions {time_range}",
            "List successful sudo commands {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} (rule.groups{{}}="authentication_success" OR rule.description="*successful*") | {aggregation}',
        variations={
            "time_range": ["in the last hour", "in the last 24 hours", "in the last 7 days"],
            "earliest": ["-1h", "-24h", "-7d"],
            "aggregation": [
                "stats count by agent.name, rule.description",
                "table _time agent.name rule.description",
                "stats count by agent.name | sort -count",
            ]
        }
    ),

    # Rule level / severity queries
    QueryTemplate(
        category="severity",
        instruction_templates=[
            "Find high severity alerts {time_range}",
            "Show me critical security alerts {time_range}",
            "Get alerts with rule level 7 or higher {time_range}",
            "List severe security events {time_range}",
            "Find high priority alerts {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} rule.level>={level} | {aggregation}',
        variations={
            "time_range": ["in the last hour", "in the last 24 hours", "in the last 7 days", "in the last 30 days"],
            "earliest": ["-1h", "-24h", "-7d", "-30d"],
            "level": ["7", "10", "12"],
            "aggregation": [
                "table _time agent.name rule.description rule.level",
                "stats count by agent.name, rule.level | sort -rule.level",
                "stats count by rule.description, rule.level | sort -rule.level",
            ]
        }
    ),

    # Agent-based queries
    QueryTemplate(
        category="agents",
        instruction_templates=[
            "Show me alerts from all agents {time_range}",
            "Count alerts per agent {time_range}",
            "Get alerts grouped by agent name {time_range}",
            "List most active agents by alert count {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} | stats count by agent.name | sort -count | head {limit}',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days", "in the last 30 days"],
            "earliest": ["-24h", "-7d", "-30d"],
            "limit": ["10", "20", "50"],
        }
    ),

    # Rule group queries
    QueryTemplate(
        category="rule_groups",
        instruction_templates=[
            "Show me all rule groups triggered {time_range}",
            "Count alerts by rule group {time_range}",
            "List most common alert types {time_range}",
            "Get distribution of alerts by category {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} | stats count by rule.groups{{}} | sort -count | head 20',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days", "in the last 30 days"],
            "earliest": ["-24h", "-7d", "-30d"],
        }
    ),

    # Specific rule queries
    QueryTemplate(
        category="specific_rules",
        instruction_templates=[
            "Find web attacks {time_range}",
            "Show me web server alerts {time_range}",
            "Get web traffic security events {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} rule.groups{{}}="web" | {aggregation}',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days"],
            "earliest": ["-24h", "-7d"],
            "aggregation": [
                "stats count by agent.name, rule.description",
                "table _time agent.name rule.description",
            ]
        }
    ),
    QueryTemplate(
        category="specific_rules",
        instruction_templates=[
            "Find system audit events {time_range}",
            "Show me audit logs {time_range}",
            "Get system security audit {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} rule.groups{{}}="audit" | {aggregation}',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days"],
            "earliest": ["-24h", "-7d"],
            "aggregation": [
                "stats count by agent.name, rule.description",
                "table _time agent.name rule.description rule.level",
            ]
        }
    ),
    QueryTemplate(
        category="specific_rules",
        instruction_templates=[
            "Find PAM module events {time_range}",
            "Show me PAM authentication events {time_range}",
            "Get PAM security alerts {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} rule.groups{{}}="pam" | {aggregation}',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days"],
            "earliest": ["-24h", "-7d"],
            "aggregation": [
                "stats count by agent.name, rule.description",
                "table _time agent.name rule.description",
            ]
        }
    ),

    # Network queries
    QueryTemplate(
        category="network",
        instruction_templates=[
            "Find network connection events {time_range}",
            "Show me network traffic alerts {time_range}",
            "Get firewall events {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} (rule.groups{{}}="firewall" OR rule.groups{{}}="network") | {aggregation}',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days"],
            "earliest": ["-24h", "-7d"],
            "aggregation": [
                "stats count by agent.name, rule.description",
                "table _time agent.name rule.description data.srcip data.dstip",
            ]
        }
    ),

    # File integrity queries
    QueryTemplate(
        category="file_integrity",
        instruction_templates=[
            "Find file integrity changes {time_range}",
            "Show me file modification events {time_range}",
            "Get file system changes {time_range}",
            "List syscheck alerts {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} rule.groups{{}}="syscheck" | {aggregation}',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days", "in the last 30 days"],
            "earliest": ["-24h", "-7d", "-30d"],
            "aggregation": [
                "stats count by agent.name, syscheck.path",
                "table _time agent.name syscheck.path syscheck.event",
                "stats count by syscheck.event | sort -count",
            ]
        }
    ),

    # Time-based queries
    QueryTemplate(
        category="timeline",
        instruction_templates=[
            "Show me alerts over time {time_range}",
            "Get alert timeline {time_range}",
            "Show hourly alert distribution {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} | timechart span={span} count by rule.level',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days"],
            "earliest": ["-24h", "-7d"],
            "span": ["1h", "15m", "4h"],
        }
    ),

    # Top/Stats queries
    QueryTemplate(
        category="statistics",
        instruction_templates=[
            "Show me the top 10 most common alerts {time_range}",
            "List the most frequent security events {time_range}",
            "Get the most triggered rules {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} | stats count by rule.description | sort -count | head 10',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days", "in the last 30 days"],
            "earliest": ["-24h", "-7d", "-30d"],
        }
    ),
    QueryTemplate(
        category="statistics",
        instruction_templates=[
            "Count unique rule IDs {time_range}",
            "List all unique rules triggered {time_range}",
            "Get distinct alert types {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} | stats dc(rule.id) as unique_rules, count as total_alerts | eval avg_per_rule=round(total_alerts/unique_rules, 2)',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days"],
            "earliest": ["-24h", "-7d"],
        }
    ),

    # MITRE ATT&CK queries
    QueryTemplate(
        category="mitre",
        instruction_templates=[
            "Find alerts with MITRE ATT&CK techniques {time_range}",
            "Show me MITRE technique coverage {time_range}",
            "Get alerts mapped to MITRE framework {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} rule.mitre.id=* | stats count by rule.mitre.id, rule.mitre.technique | sort -count',
        variations={
            "time_range": ["in the last 7 days", "in the last 30 days"],
            "earliest": ["-7d", "-30d"],
        }
    ),

    # Sudo queries
    QueryTemplate(
        category="sudo",
        instruction_templates=[
            "Find sudo command executions {time_range}",
            "Show me privilege escalation attempts {time_range}",
            "Get sudo to ROOT events {time_range}",
            "List sudo activity {time_range}",
        ],
        query_template='index=wazuh-alerts earliest={earliest} rule.groups{{}}="sudo" | {aggregation}',
        variations={
            "time_range": ["in the last 24 hours", "in the last 7 days"],
            "earliest": ["-24h", "-7d"],
            "aggregation": [
                "stats count by agent.name, rule.description",
                "table _time agent.name rule.description data.srcuser",
                "stats count by data.srcuser | sort -count",
            ]
        }
    ),
]


def splunk_search(query: str, max_wait: int = 60) -> Tuple[bool, int, Optional[str]]:
    """
    Run a Splunk search and return (success, result_count, error_message).
    """
    headers = {"Authorization": f"Bearer {SPLUNK_TOKEN}"}

    try:
        # Submit search
        resp = requests.post(
            f"{SPLUNK_HOST}/services/search/jobs",
            headers=headers,
            data={"search": f"search {query}", "output_mode": "json"},
            verify=False,
            timeout=30
        )

        if resp.status_code not in (200, 201):
            return False, 0, f"Submit error: {resp.status_code}"

        sid = resp.json().get("sid")
        if not sid:
            return False, 0, "No SID returned"

        # Wait for completion
        for _ in range(max_wait):
            time.sleep(1)
            status_resp = requests.get(
                f"{SPLUNK_HOST}/services/search/jobs/{sid}?output_mode=json",
                headers=headers,
                verify=False,
                timeout=10
            )

            if status_resp.status_code == 200:
                content = status_resp.json().get("entry", [{}])[0].get("content", {})
                dispatch_state = content.get("dispatchState", "")

                if dispatch_state == "DONE":
                    result_count = content.get("resultCount", 0)
                    return True, result_count, None
                elif dispatch_state == "FAILED":
                    messages = content.get("messages", [])
                    error_msg = messages[0].get("text", "Unknown error") if messages else "Unknown error"
                    return False, 0, error_msg

        return False, 0, "Timeout"

    except Exception as e:
        return False, 0, str(e)


def generate_query_from_template(template: QueryTemplate) -> Tuple[str, str]:
    """
    Generate an instruction and query from a template.
    Returns (instruction, query).
    """
    # Pick random instruction template
    instruction_template = random.choice(template.instruction_templates)

    # Build variations dict
    selected_vars = {}
    for var_name, var_options in template.variations.items():
        selected_vars[var_name] = random.choice(var_options)

    # Generate instruction
    instruction = instruction_template.format(**{k: v for k, v in selected_vars.items() if k in instruction_template})

    # Generate query
    query = template.query_template
    for var_name, var_value in selected_vars.items():
        query = query.replace("{" + var_name + "}", var_value)

    return instruction, query


def main():
    print("=" * 70)
    print("Wazuh Training Query Generator")
    print("=" * 70)
    print(f"Splunk Host: {SPLUNK_HOST}")
    print(f"Output File: {OUTPUT_FILE}")
    print()

    # Test Splunk connection
    print("Testing Splunk connection...")
    success, count, error = splunk_search("index=wazuh-alerts | head 1")
    if not success:
        print(f"  ERROR: Cannot connect to Splunk: {error}")
        return 1
    print(f"  OK - Splunk connected")
    print()

    # Generate queries
    validated_queries = []
    failed_queries = []

    target_count = 200  # Generate 200 validated queries
    attempts = 0
    max_attempts = 500

    print(f"Generating {target_count} validated training queries...")
    print()

    while len(validated_queries) < target_count and attempts < max_attempts:
        attempts += 1

        # Pick random template
        template = random.choice(QUERY_TEMPLATES)

        # Generate instruction and query
        instruction, query = generate_query_from_template(template)

        # Skip if we already have this exact instruction
        if any(q["instruction"] == instruction for q in validated_queries):
            continue

        # Validate query
        success, result_count, error = splunk_search(query, max_wait=30)

        if success and result_count > 0:
            validated_queries.append({
                "instruction": instruction,
                "output": query,
                "category": template.category,
                "result_count": result_count,
                "validated_at": datetime.utcnow().isoformat()
            })
            print(f"  [{len(validated_queries):3d}/{target_count}] OK ({result_count:6d} results) - {instruction[:50]}...")
        else:
            failed_queries.append({
                "instruction": instruction,
                "query": query,
                "error": error or "No results"
            })
            if len(failed_queries) % 10 == 0:
                print(f"  ... {len(failed_queries)} queries failed validation so far")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Validated queries: {len(validated_queries)}")
    print(f"Failed queries: {len(failed_queries)}")
    print(f"Attempts: {attempts}")
    print()

    # Save validated queries
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        for q in validated_queries:
            # Format for training: just instruction and output
            training_entry = {
                "instruction": q["instruction"],
                "output": q["output"]
            }
            f.write(json.dumps(training_entry) + "\n")

    print(f"Saved {len(validated_queries)} queries to {OUTPUT_FILE}")

    # Also save with metadata for reference
    metadata_file = OUTPUT_FILE.replace(".jsonl", "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "total_queries": len(validated_queries),
            "categories": list(set(q["category"] for q in validated_queries)),
            "queries": validated_queries
        }, f, indent=2)

    print(f"Saved metadata to {metadata_file}")

    # Save failed queries for debugging
    if failed_queries:
        failed_file = OUTPUT_FILE.replace(".jsonl", "_failed.json")
        with open(failed_file, "w") as f:
            json.dump(failed_queries[:50], f, indent=2)  # Save first 50
        print(f"Saved {min(50, len(failed_queries))} failed queries to {failed_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
