#!/usr/bin/env python3
"""
Fetch Wazuh field mappings and sample data from Splunk for RAG system.
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
SPLUNK_HOST = "https://10.10.10.114:8089"
SPLUNK_TOKEN = "eyJraWQiOiJzcGx1bmsuc2VjcmV0IiwiYWxnIjoiSFM1MTIiLCJ2ZXIiOiJ2MiIsInR0eXAiOiJzdGF0aWMifQ.eyJpc3MiOiJhZmVybmFuZGV6IGZyb20gc3BzZWFyY2giLCJzdWIiOiJhZmVybmFuZGV6IiwiYXVkIjoiSW1wcm92aW5nIFpldXMgdG9vbCIsImlkcCI6IkxEQVA6Ly9PcGVuTERBUCIsImp0aSI6IjliODJmYWU1OTg2Y2YyZjhkZDczMWY4MWVlYTg5YWRkMjE0MGQ2MGQ2OGY3YTJhZTNjMjMxMmQ2MGEwNzQ2OTUiLCJpYXQiOjE3NjUyMjA4MzgsImV4cCI6MTc2NzgxMjgzOCwibmJyIjoxNzY1MjIwODM4fQ.sxGtJMsayNdKNW8wj9zeT2Yf0Y9yMmulLgsPL3QrulzvAsfchS0qgZTnAnMcaUa5sirQowQp8fFhfAkLah07bg"

# Disable SSL warnings for self-signed certs
requests.packages.urllib3.disable_warnings()

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "wazuh_rag"


def run_search(query: str, max_wait: int = 120) -> dict:
    """Submit a search job and wait for results."""
    headers = {"Authorization": f"Bearer {SPLUNK_TOKEN}"}

    # Submit search
    print(f"  Submitting: {query[:80]}...")
    resp = requests.post(
        f"{SPLUNK_HOST}/services/search/jobs",
        headers=headers,
        data={"search": query, "output_mode": "json"},
        verify=False
    )

    if resp.status_code != 201 and resp.status_code != 200:
        print(f"  Error submitting search: {resp.status_code} - {resp.text}")
        return {"results": []}

    data = resp.json()
    sid = data.get("sid")
    if not sid:
        print(f"  No SID returned: {data}")
        return {"results": []}

    print(f"  Job ID: {sid}")

    # Wait for job to complete
    for i in range(max_wait):
        time.sleep(1)
        status_resp = requests.get(
            f"{SPLUNK_HOST}/services/search/jobs/{sid}?output_mode=json",
            headers=headers,
            verify=False
        )
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            dispatch_state = status_data.get("entry", [{}])[0].get("content", {}).get("dispatchState", "")
            if dispatch_state == "DONE":
                print(f"  Job completed in {i+1}s")
                break
            elif dispatch_state == "FAILED":
                print(f"  Job failed")
                return {"results": []}
        if i % 10 == 0 and i > 0:
            print(f"  Still waiting... ({i}s)")

    # Get results
    results_resp = requests.get(
        f"{SPLUNK_HOST}/services/search/jobs/{sid}/results?output_mode=json&count=10000",
        headers=headers,
        verify=False
    )

    if results_resp.status_code == 200:
        return results_resp.json()

    print(f"  Error getting results: {results_resp.status_code}")
    return {"results": []}


def main():
    print("=" * 60)
    print("Fetching Wazuh Field Mappings from Splunk")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {
        "fields": {},
        "rule_ids": [],
        "rule_groups": [],
        "sample_events": [],
        "top_queries": []
    }

    # 1. Get all field summary
    print("\n[1/6] Getting all field summary...")
    result = run_search(
        "search index=wazuh-alerts earliest=-30d | fieldsummary | fields field count distinct_count"
    )
    all_data["fields"]["all"] = result.get("results", [])
    print(f"  Found {len(all_data['fields']['all'])} fields")

    # 2. Get agent/rule/data fields specifically
    print("\n[2/6] Getting agent/rule/data fields...")
    result = run_search(
        'search index=wazuh-alerts earliest=-30d | fieldsummary | where match(field, "^(agent|rule|data|decoder|manager|location|predecoder)") | fields field count distinct_count'
    )
    all_data["fields"]["core"] = result.get("results", [])
    print(f"  Found {len(all_data['fields']['core'])} core fields")

    # 3. Get unique rule IDs and descriptions
    print("\n[3/6] Getting rule IDs and descriptions...")
    result = run_search(
        'search index=wazuh-alerts earliest=-30d | stats count by rule.id, rule.description, rule.level | sort -count | head 200'
    )
    all_data["rule_ids"] = result.get("results", [])
    print(f"  Found {len(all_data['rule_ids'])} unique rules")

    # 4. Get unique rule groups
    print("\n[4/6] Getting rule groups...")
    result = run_search(
        'search index=wazuh-alerts earliest=-30d | stats count by rule.groups{} | sort -count | head 100'
    )
    all_data["rule_groups"] = result.get("results", [])
    print(f"  Found {len(all_data['rule_groups'])} rule groups")

    # 5. Get sample events (diverse)
    print("\n[5/6] Getting sample events...")
    result = run_search(
        'search index=wazuh-alerts earliest=-7d | dedup rule.id | head 50 | fields agent.*, rule.*, data.*, decoder.*, location, manager.*, timestamp'
    )
    all_data["sample_events"] = result.get("results", [])
    print(f"  Got {len(all_data['sample_events'])} sample events")

    # 6. Get top query patterns (based on common searches)
    print("\n[6/6] Getting MITRE ATT&CK mappings...")
    result = run_search(
        'search index=wazuh-alerts earliest=-30d rule.mitre.id{}=* | stats count by rule.mitre.id{}, rule.mitre.technique{}, rule.mitre.tactic{} | sort -count | head 50'
    )
    all_data["mitre_mappings"] = result.get("results", [])
    print(f"  Found {len(all_data.get('mitre_mappings', []))} MITRE mappings")

    # Save all data
    output_file = OUTPUT_DIR / "wazuh_field_data.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n✓ Saved all data to {output_file}")

    # Create a summary file for quick reference
    summary = {
        "total_fields": len(all_data["fields"].get("all", [])),
        "core_fields": len(all_data["fields"].get("core", [])),
        "unique_rules": len(all_data["rule_ids"]),
        "rule_groups": len(all_data["rule_groups"]),
        "sample_events": len(all_data["sample_events"]),
        "mitre_mappings": len(all_data.get("mitre_mappings", []))
    }

    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total fields: {summary['total_fields']}")
    print(f"  Core fields: {summary['core_fields']}")
    print(f"  Unique rules: {summary['unique_rules']}")
    print(f"  Rule groups: {summary['rule_groups']}")
    print(f"  Sample events: {summary['sample_events']}")
    print(f"  MITRE mappings: {summary['mitre_mappings']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
