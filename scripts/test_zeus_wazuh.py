#!/usr/bin/env python3
"""
Test Zeus Wazuh query generation against actual Splunk data.
"""

import requests
import json
import time
import sys

# Disable SSL warnings
requests.packages.urllib3.disable_warnings()

# Configuration
ZEUS_URL = "http://localhost:8081/api"
SPLUNK_HOST = "https://10.10.10.114:8089"
SPLUNK_TOKEN = "eyJraWQiOiJzcGx1bmsuc2VjcmV0IiwiYWxnIjoiSFM1MTIiLCJ2ZXIiOiJ2MiIsInR0eXAiOiJzdGF0aWMifQ.eyJpc3MiOiJhZmVybmFuZGV6IGZyb20gc3BzZWFyY2giLCJzdWIiOiJhZmVybmFuZGV6IiwiYXVkIjoiSW1wcm92aW5nIFpldXMgdG9vbCIsImlkcCI6IkxEQVA6Ly9PcGVuTERBUCIsImp0aSI6IjliODJmYWU1OTg2Y2YyZjhkZDczMWY4MWVlYTg5YWRkMjE0MGQ2MGQ2OGY3YTJhZTNjMjMxMmQ2MGEwNzQ2OTUiLCJpYXQiOjE3NjUyMjA4MzgsImV4cCI6MTc2NzgxMjgzOCwibmJyIjoxNzY1MjIwODM4fQ.sxGtJMsayNdKNW8wj9zeT2Yf0Y9yMmulLgsPL3QrulzvAsfchS0qgZTnAnMcaUa5sirQowQp8fFhfAkLah07bg"

# Test user credentials
ZEUS_USER = "zeustest"
ZEUS_PASS = "testpass123"


def zeus_login():
    """Login to Zeus and get auth token."""
    print("Logging into Zeus...")
    try:
        resp = requests.post(
            f"{ZEUS_URL}/auth/login",
            json={"username": ZEUS_USER, "password": ZEUS_PASS},
            timeout=30
        )
        if resp.status_code == 200:
            token = resp.json().get("access_token")
            print(f"  ✓ Logged in successfully")
            return token
        else:
            print(f"  ✗ Login failed: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"  ✗ Login error: {e}")
        return None


def zeus_generate(token: str, instruction: str):
    """Generate a query through Zeus."""
    print(f"\nGenerating query for: '{instruction}'")
    try:
        resp = requests.post(
            f"{ZEUS_URL}/generate",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "instruction": instruction,
                "indexes": ["wazuh-alerts"],
                "num_return_sequences": 1,
                "temperature": 0.3
            },
            timeout=120
        )
        if resp.status_code == 200:
            data = resp.json()
            query = data.get("query", "")
            print(f"  Generated: {query}")
            return query
        else:
            print(f"  ✗ Generation failed: {resp.status_code} - {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  ✗ Generation error: {e}")
        return None


def splunk_search(query: str, max_wait: int = 60):
    """Run a search on Splunk and return result count."""
    headers = {"Authorization": f"Bearer {SPLUNK_TOKEN}"}

    # Add time constraint if not present - must go right after index=
    if "earliest=" not in query.lower():
        # Insert earliest right after the index clause
        import re
        query = re.sub(r'(index=\S+)', r'\1 earliest=-30d', query, count=1)
    
    print(f"  Running on Splunk: {query[:100]}...")
    
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
            print(f"  ✗ Splunk submit error: {resp.status_code} - {resp.text[:200]}")
            return None, "SUBMIT_ERROR"
        
        sid = resp.json().get("sid")
        if not sid:
            print(f"  ✗ No SID returned")
            return None, "NO_SID"
        
        # Wait for completion
        for i in range(max_wait):
            time.sleep(1)
            status_resp = requests.get(
                f"{SPLUNK_HOST}/services/search/jobs/{sid}?output_mode=json",
                headers=headers,
                verify=False,
                timeout=10
            )
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                entry = status_data.get("entry", [{}])[0]
                content = entry.get("content", {})
                dispatch_state = content.get("dispatchState", "")
                
                if dispatch_state == "DONE":
                    result_count = content.get("resultCount", 0)
                    event_count = content.get("eventCount", 0)
                    break
                elif dispatch_state == "FAILED":
                    messages = content.get("messages", [])
                    error_msg = messages[0].get("text", "Unknown error") if messages else "Unknown error"
                    print(f"  ✗ Search failed: {error_msg[:100]}")
                    return None, f"FAILED: {error_msg[:50]}"
            
            if i == max_wait - 1:
                print(f"  ✗ Search timed out")
                return None, "TIMEOUT"
        
        # Get results
        results_resp = requests.get(
            f"{SPLUNK_HOST}/services/search/jobs/{sid}/results?output_mode=json&count=5",
            headers=headers,
            verify=False,
            timeout=30
        )
        
        if results_resp.status_code == 200:
            results = results_resp.json().get("results", [])
            return result_count, results
        
        return result_count, []
        
    except Exception as e:
        print(f"  ✗ Splunk error: {e}")
        return None, str(e)


def test_query(token: str, instruction: str, expected_fields: list = None):
    """Test a single query end-to-end."""
    print("\n" + "=" * 70)
    
    # Generate query
    query = zeus_generate(token, instruction)
    if not query:
        return {"instruction": instruction, "status": "GENERATION_FAILED", "query": None, "count": 0}
    
    # Run on Splunk
    count, results = splunk_search(query)
    
    if count is None:
        status = f"SPLUNK_ERROR: {results}"
        print(f"  ✗ Status: {status}")
    elif count == 0:
        status = "NO_RESULTS"
        print(f"  ✗ Status: No results returned")
    else:
        status = "SUCCESS"
        print(f"  ✓ Status: {count} results returned")
        if results and isinstance(results, list) and len(results) > 0:
            print(f"  Sample result keys: {list(results[0].keys())[:10]}")
    
    return {
        "instruction": instruction,
        "query": query,
        "status": status,
        "count": count if count else 0,
        "sample": results[0] if results and isinstance(results, list) and len(results) > 0 else None
    }


def main():
    # Login to Zeus
    token = zeus_login()
    if not token:
        print("Failed to login. Check credentials.")
        sys.exit(1)
    
    # Test queries based on actual Wazuh data we know exists
    test_cases = [
        "Show me all alerts from the last 7 days",
        "Count alerts by agent name",
        "Show failed authentication attempts by agent",  # Don't request data.srcip - not in auth events
        "Find alerts with rule level 7 or higher",  # Max level in data is 7
        "Show web traffic alerts",
        "List unique rule IDs and their descriptions",
        "Find sudo command execution alerts",
        "Show alerts grouped by rule group",
        "Find authentication success events",
        "Show network connections with source and destination IPs",
    ]
    
    results = []
    for instruction in test_cases:
        result = test_query(token, instruction)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = len(results) - success
    
    print(f"\nTotal: {len(results)} | Success: {success} | Failed: {failed}")
    print(f"Success Rate: {success/len(results)*100:.1f}%\n")
    
    print("Results by query:")
    for r in results:
        status_icon = "✓" if r["status"] == "SUCCESS" else "✗"
        print(f"  {status_icon} [{r['count']:>5}] {r['instruction'][:50]}")
        if r["status"] != "SUCCESS":
            print(f"           Status: {r['status']}")
            if r["query"]:
                print(f"           Query: {r['query'][:80]}...")
    
    # Save detailed results
    with open("/home/purpleforge/25tools/Zeus/data/wazuh_rag/test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to data/wazuh_rag/test_results.json")


if __name__ == "__main__":
    main()
