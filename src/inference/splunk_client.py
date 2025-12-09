"""
Splunk API Client for Zeus.

Provides functionality to:
- Connect to Splunk REST API
- Fetch available indexes and their metadata
- Run search queries for validation
- Generate training data from real log data
"""

import os
import re
import time
import json
import requests
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger

# Disable SSL warnings for self-signed certs
requests.packages.urllib3.disable_warnings()


@dataclass
class SplunkConfig:
    """Splunk connection configuration."""
    host: str
    token: str
    verify_ssl: bool = False
    timeout: int = 30


@dataclass
class IndexMetadata:
    """Metadata about a Splunk index."""
    name: str
    event_count: int
    earliest_time: Optional[str]
    latest_time: Optional[str]
    common_fields: List[str]
    sourcetypes: List[str]


@dataclass
class SearchResult:
    """Result of a Splunk search."""
    success: bool
    result_count: int
    event_count: int
    error_message: Optional[str]
    sample_results: List[Dict[str, Any]]
    execution_time: float


@dataclass
class DataCapabilityMismatch:
    """Represents a mismatch between requested data and available data sources."""
    requested_capability: str
    requested_description: str
    available_indexes: List[str]
    missing_data_type: str
    suggestion: str
    severity: str  # "warning" or "error"


class SplunkClient:
    """Client for interacting with Splunk REST API."""

    # Data source capabilities - what each index type CAN provide
    INDEX_CAPABILITIES = {
        "wazuh-alerts": {
            "type": "EDR/Host Security",
            "description": "Wazuh EDR security alerts from host agents",
            "provides": [
                "authentication_events",      # Login success/failure
                "file_integrity",             # File changes (syscheck)
                "process_alerts",             # Suspicious process activity (alerts only, not full telemetry)
                "vulnerability_detection",    # CVE detections
                "compliance_checks",          # CIS, PCI-DSS checks
                "rootkit_detection",          # Rootcheck alerts
                "system_events",              # System-level alerts
                "web_attacks",                # Web attack detection (alerts)
                "agent_status",               # Agent health/status
            ],
            "does_not_provide": [
                "full_network_traffic",       # No NetFlow/packet data
                "dns_queries",                # No DNS query logs (unless forwarded)
                "proxy_logs",                 # No web proxy logs
                "email_logs",                 # No email server logs
                "firewall_traffic",           # No firewall allow/deny logs
                "full_process_telemetry",     # Only alerts, not all process starts
                "network_connections",        # No connection-level data
                "bandwidth_metrics",          # No throughput data
                "packet_captures",            # No PCAP data
            ],
            "field_examples": {
                "agent.name": "Hostname of the agent",
                "rule.description": "Alert description",
                "rule.level": "Severity (0-15)",
                "data.srcip": "Source IP (when available in alert)",
                "syscheck.path": "File path for FIM events",
            }
        },
        "network_traffic": {
            "type": "Network Traffic",
            "description": "Full network flow and connection data",
            "provides": [
                "full_network_traffic",
                "network_connections",
                "dns_queries",
                "bandwidth_metrics",
                "port_scans",
                "lateral_movement",
            ],
            "does_not_provide": [
                "process_execution",
                "file_integrity",
                "authentication_events",
            ],
            "field_examples": {
                "src_ip": "Source IP address",
                "dest_ip": "Destination IP address",
                "bytes_in": "Inbound bytes",
                "bytes_out": "Outbound bytes",
            }
        },
        "firewall": {
            "type": "Firewall Logs",
            "description": "Firewall allow/deny decisions and traffic logs",
            "provides": [
                "firewall_traffic",
                "blocked_connections",
                "allowed_connections",
                "port_access",
            ],
            "does_not_provide": [
                "process_execution",
                "file_integrity",
                "authentication_events",
                "full_packet_content",
            ],
            "field_examples": {
                "action": "allow/deny",
                "src_ip": "Source IP",
                "dest_port": "Destination port",
            }
        },
        "proxy": {
            "type": "Web Proxy",
            "description": "Web proxy/gateway logs",
            "provides": [
                "proxy_logs",
                "web_traffic",
                "url_access",
                "user_web_activity",
            ],
            "does_not_provide": [
                "process_execution",
                "file_integrity",
                "non_http_traffic",
            ],
            "field_examples": {
                "url": "Requested URL",
                "user": "Username",
                "bytes": "Transfer size",
            }
        },
        "sysmon": {
            "type": "Sysmon/Process Telemetry",
            "description": "Detailed process and system activity",
            "provides": [
                "full_process_telemetry",
                "process_creation",
                "network_connections",
                "file_creation",
                "registry_changes",
                "dns_queries",
            ],
            "does_not_provide": [
                "full_network_traffic",
                "packet_content",
                "bandwidth_metrics",
            ],
            "field_examples": {
                "Image": "Process path",
                "CommandLine": "Full command line",
                "ParentImage": "Parent process",
            }
        },
    }

    # What types of queries require what data capabilities
    QUERY_REQUIREMENTS = {
        "network_traffic_analysis": {
            "description": "Analyzing network traffic, connections, or bandwidth",
            "keywords": [
                "traffic", "bandwidth", "netflow", "connection", "bytes transferred",
                "data transfer", "network flow", "port 443", "port 80", "port 22",
                "443 traffic", "80 traffic", "22 traffic", "https traffic", "http traffic",
                "ssh traffic", "rdp traffic", "port 3389", "all traffic", "inbound traffic",
                "outbound traffic", "network traffic", "tcp traffic", "udp traffic",
                "dest_port", "src_port", "bytes_in", "bytes_out"
            ],
            "requires": ["full_network_traffic", "network_connections"],
            "alternative_with_edr": "Wazuh EDR shows security ALERTS (like 'possible port scan detected') but NOT actual traffic data. For traffic on port 443/80/etc, you need NetFlow, Zeek, or firewall logs."
        },
        "dns_analysis": {
            "description": "DNS query analysis and threat hunting",
            "keywords": ["dns", "domain lookup", "name resolution", "dns query", "dns request"],
            "requires": ["dns_queries"],
            "alternative_with_edr": "Wazuh doesn't capture DNS queries directly. You need DNS logs from your DNS server, Sysmon with DNS logging, or network monitoring tools."
        },
        "firewall_analysis": {
            "description": "Firewall rule hits, blocked traffic analysis",
            "keywords": ["firewall", "blocked", "allowed", "acl", "rule hit", "deny", "permit"],
            "requires": ["firewall_traffic"],
            "alternative_with_edr": "Wazuh can detect some network-related threats but doesn't have firewall allow/deny logs. Check your firewall's logs directly."
        },
        "proxy_analysis": {
            "description": "Web proxy and user browsing analysis",
            "keywords": ["proxy", "web browsing", "url access", "web gateway", "user browsing"],
            "requires": ["proxy_logs"],
            "alternative_with_edr": "Wazuh doesn't have web proxy data. For user browsing analysis, you need proxy logs (Squid, Zscaler, etc.)."
        },
        "full_process_telemetry": {
            "description": "Complete process execution history",
            "keywords": ["all processes", "process history", "every process", "process tree", "all executions"],
            "requires": ["full_process_telemetry"],
            "alternative_with_edr": "Wazuh provides process-related ALERTS (suspicious activity) but not a complete log of every process. For full telemetry, you need Sysmon or EDR agents with process logging."
        },
        "email_analysis": {
            "description": "Email traffic and phishing analysis",
            "keywords": ["email", "phishing", "mail", "smtp", "inbox", "attachment"],
            "requires": ["email_logs"],
            "alternative_with_edr": "Wazuh doesn't have email logs. For email analysis, you need logs from your email gateway or mail server."
        },
        "packet_analysis": {
            "description": "Deep packet inspection and content analysis",
            "keywords": ["packet", "payload", "pcap", "deep packet", "packet content", "raw traffic"],
            "requires": ["packet_captures"],
            "alternative_with_edr": "Wazuh doesn't capture packet content. For packet analysis, you need PCAP data from network taps or IDS sensors."
        },
    }

    # Log type categories for analyst context
    LOG_CATEGORIES = {
        "authentication": {
            "description": "Login attempts, authentication events, access control",
            "keywords": ["login", "logout", "auth", "password", "credential", "session", "sudo", "ssh"],
            "rule_groups": ["authentication_failed", "authentication_success", "pam", "sshd", "sudo"],
            "example_questions": [
                "Which user accounts are you investigating?",
                "Are you looking for failed or successful authentications?",
                "What time period should I search?",
                "Are you investigating a specific host or all hosts?"
            ]
        },
        "network": {
            "description": "Network connections, traffic, firewall events",
            "keywords": ["connection", "network", "firewall", "traffic", "ip", "port", "dns", "http"],
            "rule_groups": ["network", "firewall", "ids", "web"],
            "example_questions": [
                "Are you looking for inbound or outbound connections?",
                "Do you have specific IP addresses to investigate?",
                "What ports or protocols are relevant?",
                "Are you looking for blocked or allowed traffic?"
            ]
        },
        "process": {
            "description": "Process execution, command line activity, scripts",
            "keywords": ["process", "command", "execute", "script", "powershell", "bash", "cmd"],
            "rule_groups": ["process", "sysmon", "ossec"],
            "example_questions": [
                "What process or command are you looking for?",
                "Are you investigating a specific user's activity?",
                "Do you want to see parent-child process relationships?",
                "What time period should I search?"
            ]
        },
        "file": {
            "description": "File operations, integrity monitoring, changes",
            "keywords": ["file", "modify", "create", "delete", "change", "integrity", "permission"],
            "rule_groups": ["syscheck", "fim", "audit"],
            "example_questions": [
                "What file paths are you interested in?",
                "Are you looking for specific file types?",
                "What type of file operations (create/modify/delete)?",
                "Are you tracking specific user activity?"
            ]
        },
        "malware": {
            "description": "Malware detection, threat indicators, suspicious activity",
            "keywords": ["malware", "virus", "trojan", "suspicious", "threat", "ioc", "hash"],
            "rule_groups": ["rootcheck", "vulnerability-detector", "virustotal"],
            "example_questions": [
                "What indicators or signatures are you looking for?",
                "Are you investigating a known threat or hunting?",
                "Do you have specific file hashes or IP addresses?",
                "What hosts should I search?"
            ]
        },
        "compliance": {
            "description": "Compliance monitoring, policy violations, audit",
            "keywords": ["compliance", "policy", "cis", "pci", "hipaa", "audit", "benchmark"],
            "rule_groups": ["cis", "pci_dss", "hipaa", "gdpr", "policy_monitoring"],
            "example_questions": [
                "Which compliance framework (CIS, PCI-DSS, HIPAA)?",
                "Are you looking for passed or failed checks?",
                "What specific controls or benchmarks?",
                "Which systems should I check?"
            ]
        },
        "web": {
            "description": "Web server logs, HTTP traffic, web attacks",
            "keywords": ["http", "web", "url", "request", "response", "apache", "nginx", "attack"],
            "rule_groups": ["web", "apache", "nginx", "attack"],
            "example_questions": [
                "What HTTP methods are relevant (GET, POST, etc.)?",
                "Are you looking for specific status codes?",
                "Do you have URLs or paths to investigate?",
                "Are you looking for attack patterns?"
            ]
        },
        "system": {
            "description": "System events, services, errors, performance",
            "keywords": ["system", "service", "error", "warning", "boot", "shutdown", "performance"],
            "rule_groups": ["system", "ossec", "syslog"],
            "example_questions": [
                "What type of system events (errors, warnings, info)?",
                "Are you investigating specific services?",
                "What hosts or systems are relevant?",
                "What time period should I search?"
            ]
        }
    }

    def __init__(self, config: Optional[SplunkConfig] = None):
        """Initialize Splunk client with configuration."""
        if config:
            self.config = config
        else:
            # Load from environment
            self.config = SplunkConfig(
                host=os.getenv("SPLUNK_HOST", "https://10.10.10.114:8089"),
                token=os.getenv("SPLUNK_TOKEN", ""),
                verify_ssl=os.getenv("SPLUNK_VERIFY_SSL", "false").lower() == "true",
                timeout=int(os.getenv("SPLUNK_TIMEOUT", "30"))
            )

        self._headers = {"Authorization": f"Bearer {self.config.token}"}
        self._indexes_cache: Optional[List[IndexMetadata]] = None
        self._cache_time: Optional[float] = None
        self._cache_ttl = 3600  # Cache for 1 hour

    def is_configured(self) -> bool:
        """Check if Splunk client is properly configured."""
        return bool(self.config.host and self.config.token)

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to Splunk."""
        if not self.is_configured():
            return False, "Splunk not configured - missing host or token"

        try:
            resp = requests.get(
                f"{self.config.host}/services/server/info?output_mode=json",
                headers=self._headers,
                verify=self.config.verify_ssl,
                timeout=self.config.timeout
            )
            if resp.status_code == 200:
                info = resp.json()
                version = info.get("entry", [{}])[0].get("content", {}).get("version", "unknown")
                return True, f"Connected to Splunk {version}"
            else:
                return False, f"Connection failed: {resp.status_code}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def get_available_indexes(self, force_refresh: bool = False) -> List[IndexMetadata]:
        """Get list of available indexes with metadata."""
        # Check cache
        if not force_refresh and self._indexes_cache and self._cache_time:
            if time.time() - self._cache_time < self._cache_ttl:
                return self._indexes_cache

        if not self.is_configured():
            logger.warning("Splunk not configured, returning default indexes")
            return self._get_default_indexes()

        try:
            # Get index list
            resp = requests.get(
                f"{self.config.host}/services/data/indexes?output_mode=json&count=0",
                headers=self._headers,
                verify=self.config.verify_ssl,
                timeout=self.config.timeout
            )

            if resp.status_code != 200:
                logger.error(f"Failed to get indexes: {resp.status_code}")
                return self._get_default_indexes()

            indexes = []
            for entry in resp.json().get("entry", []):
                name = entry.get("name", "")
                content = entry.get("content", {})

                # Skip internal indexes unless specifically needed
                if name.startswith("_") and name not in ["_audit", "_internal"]:
                    continue

                indexes.append(IndexMetadata(
                    name=name,
                    event_count=content.get("totalEventCount", 0),
                    earliest_time=content.get("minTime"),
                    latest_time=content.get("maxTime"),
                    common_fields=[],  # Will be populated lazily
                    sourcetypes=[]  # Will be populated lazily
                ))

            # Sort by event count (most active first)
            indexes.sort(key=lambda x: x.event_count, reverse=True)

            self._indexes_cache = indexes
            self._cache_time = time.time()

            return indexes

        except Exception as e:
            logger.error(f"Error getting indexes: {e}")
            return self._get_default_indexes()

    def _get_default_indexes(self) -> List[IndexMetadata]:
        """Return default indexes when Splunk is not available."""
        return [
            IndexMetadata(
                name="wazuh-alerts",
                event_count=0,
                earliest_time=None,
                latest_time=None,
                common_fields=["rule.description", "rule.level", "agent.name", "data.srcip"],
                sourcetypes=["wazuh_alerts"]
            ),
            IndexMetadata(
                name="main",
                event_count=0,
                earliest_time=None,
                latest_time=None,
                common_fields=[],
                sourcetypes=[]
            )
        ]

    def get_index_fields(self, index_name: str, sample_size: int = 1000) -> List[str]:
        """Get common fields for an index by sampling events."""
        if not self.is_configured():
            return []

        try:
            query = f"index={index_name} | head {sample_size} | fieldsummary | table field"
            result = self.run_search(query, max_wait=60)

            if result.success and result.sample_results:
                return [r.get("field", "") for r in result.sample_results if r.get("field")]
            return []

        except Exception as e:
            logger.error(f"Error getting fields for {index_name}: {e}")
            return []

    def get_index_sourcetypes(self, index_name: str) -> List[str]:
        """Get sourcetypes available in an index."""
        if not self.is_configured():
            return []

        try:
            query = f"index={index_name} earliest=-7d | stats count by sourcetype | sort -count | head 20"
            result = self.run_search(query, max_wait=60)

            if result.success and result.sample_results:
                return [r.get("sourcetype", "") for r in result.sample_results if r.get("sourcetype")]
            return []

        except Exception as e:
            logger.error(f"Error getting sourcetypes for {index_name}: {e}")
            return []

    def run_search(self, query: str, max_wait: int = 60, max_results: int = 100) -> SearchResult:
        """Run a Splunk search and return results."""
        start_time = time.time()

        if not self.is_configured():
            return SearchResult(
                success=False,
                result_count=0,
                event_count=0,
                error_message="Splunk not configured",
                sample_results=[],
                execution_time=0
            )

        # Ensure query has time constraints
        if "earliest=" not in query.lower():
            query = re.sub(r'(index=\S+)', r'\1 earliest=-30d', query, count=1)

        try:
            # Submit search job
            resp = requests.post(
                f"{self.config.host}/services/search/jobs",
                headers=self._headers,
                data={"search": f"search {query}", "output_mode": "json"},
                verify=self.config.verify_ssl,
                timeout=self.config.timeout
            )

            if resp.status_code not in (200, 201):
                return SearchResult(
                    success=False,
                    result_count=0,
                    event_count=0,
                    error_message=f"Submit error: {resp.status_code} - {resp.text[:200]}",
                    sample_results=[],
                    execution_time=time.time() - start_time
                )

            sid = resp.json().get("sid")
            if not sid:
                return SearchResult(
                    success=False,
                    result_count=0,
                    event_count=0,
                    error_message="No search ID returned",
                    sample_results=[],
                    execution_time=time.time() - start_time
                )

            # Wait for completion
            result_count = 0
            event_count = 0
            for _ in range(max_wait):
                time.sleep(1)
                status_resp = requests.get(
                    f"{self.config.host}/services/search/jobs/{sid}?output_mode=json",
                    headers=self._headers,
                    verify=self.config.verify_ssl,
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
                        return SearchResult(
                            success=False,
                            result_count=0,
                            event_count=0,
                            error_message=f"Search failed: {error_msg}",
                            sample_results=[],
                            execution_time=time.time() - start_time
                        )
            else:
                return SearchResult(
                    success=False,
                    result_count=0,
                    event_count=0,
                    error_message="Search timed out",
                    sample_results=[],
                    execution_time=time.time() - start_time
                )

            # Get results
            results_resp = requests.get(
                f"{self.config.host}/services/search/jobs/{sid}/results?output_mode=json&count={max_results}",
                headers=self._headers,
                verify=self.config.verify_ssl,
                timeout=30
            )

            sample_results = []
            if results_resp.status_code == 200:
                sample_results = results_resp.json().get("results", [])

            return SearchResult(
                success=True,
                result_count=result_count,
                event_count=event_count,
                error_message=None,
                sample_results=sample_results,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return SearchResult(
                success=False,
                result_count=0,
                event_count=0,
                error_message=str(e),
                sample_results=[],
                execution_time=time.time() - start_time
            )

    def validate_query(self, query: str) -> Tuple[bool, str, int]:
        """
        Validate a query by running it against Splunk.

        Returns:
            (is_valid, message, result_count)
        """
        result = self.run_search(query, max_wait=30, max_results=5)

        if not result.success:
            return False, result.error_message or "Unknown error", 0

        if result.result_count == 0:
            return True, "Query executed but returned no results", 0

        return True, f"Query returned {result.result_count} results", result.result_count

    def get_log_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get available log categories for context gathering."""
        return self.LOG_CATEGORIES

    def detect_category(self, instruction: str) -> Optional[str]:
        """Detect the most likely log category from user instruction."""
        instruction_lower = instruction.lower()

        best_category = None
        best_score = 0

        for category, info in self.LOG_CATEGORIES.items():
            score = 0
            for keyword in info["keywords"]:
                if keyword in instruction_lower:
                    score += 1

            if score > best_score:
                best_score = score
                best_category = category

        return best_category if best_score > 0 else None

    def get_clarification_questions(self, instruction: str, category: Optional[str] = None) -> List[str]:
        """Get relevant clarification questions based on instruction and category."""
        if not category:
            category = self.detect_category(instruction)

        questions = []

        # Generic questions for all vague queries
        instruction_lower = instruction.lower()

        # Check for missing time context
        if not any(t in instruction_lower for t in ["hour", "day", "week", "month", "year", "last", "recent", "today", "yesterday"]):
            questions.append("What time range should I search? (e.g., last 24 hours, last 7 days)")

        # Check for missing target specificity
        if not any(t in instruction_lower for t in ["all", "every", "specific", "host", "agent", "user", "ip"]):
            questions.append("Should I search all hosts/agents or specific ones?")

        # Add category-specific questions
        if category and category in self.LOG_CATEGORIES:
            cat_questions = self.LOG_CATEGORIES[category]["example_questions"]
            # Add questions that aren't already answered in the instruction
            for q in cat_questions:
                # Simple heuristic: add if key terms not in instruction
                key_terms = ["user", "host", "time", "specific", "type", "looking"]
                if not any(term in instruction_lower for term in q.lower().split() if len(term) > 4):
                    questions.append(q)
                    if len(questions) >= 4:
                        break

        return questions[:4]  # Limit to 4 questions

    def check_data_capability(self, instruction: str, selected_indexes: List[str]) -> List[DataCapabilityMismatch]:
        """
        Check if the user's request can be fulfilled by the selected indexes.

        Returns a list of mismatches where the user is asking for data
        that the selected indexes cannot provide.
        """
        mismatches = []
        instruction_lower = instruction.lower()

        # Determine what capabilities the selected indexes provide
        available_capabilities = set()
        unavailable_capabilities = set()

        for index in selected_indexes:
            # Normalize index name
            index_key = index.lower().replace("-", "_")

            # Check if we know this index type
            if index_key in self.INDEX_CAPABILITIES:
                cap_info = self.INDEX_CAPABILITIES[index_key]
                available_capabilities.update(cap_info["provides"])
                unavailable_capabilities.update(cap_info["does_not_provide"])
            elif "wazuh" in index_key:
                # Default to wazuh-alerts capabilities for any wazuh index
                cap_info = self.INDEX_CAPABILITIES["wazuh-alerts"]
                available_capabilities.update(cap_info["provides"])
                unavailable_capabilities.update(cap_info["does_not_provide"])

        # Check each query requirement against the instruction
        for req_type, req_info in self.QUERY_REQUIREMENTS.items():
            # Check if any keywords match the instruction
            keyword_matches = [kw for kw in req_info["keywords"] if kw in instruction_lower]

            if keyword_matches:
                # User is asking for this type of data
                required_caps = set(req_info["requires"])

                # Check if any required capability is missing
                missing = required_caps - available_capabilities

                if missing and (required_caps & unavailable_capabilities):
                    # This is a real mismatch - user asking for something their indexes can't provide
                    mismatches.append(DataCapabilityMismatch(
                        requested_capability=req_type.replace("_", " ").title(),
                        requested_description=req_info["description"],
                        available_indexes=selected_indexes,
                        missing_data_type=", ".join(missing),
                        suggestion=req_info["alternative_with_edr"],
                        severity="warning" if len(missing) < len(required_caps) else "error"
                    ))

        return mismatches

    def get_index_capabilities(self, index_name: str) -> Dict[str, Any]:
        """Get capability information for a specific index."""
        index_key = index_name.lower().replace("-", "_")

        if index_key in self.INDEX_CAPABILITIES:
            return self.INDEX_CAPABILITIES[index_key]
        elif "wazuh" in index_key:
            return self.INDEX_CAPABILITIES["wazuh-alerts"]
        else:
            return {
                "type": "Unknown",
                "description": f"Unknown index type: {index_name}",
                "provides": [],
                "does_not_provide": [],
                "field_examples": {}
            }

    def get_data_source_explanation(self, selected_indexes: List[str]) -> str:
        """
        Generate an explanation of what data is available from the selected indexes.
        """
        explanations = []

        for index in selected_indexes:
            cap_info = self.get_index_capabilities(index)

            provides_text = ", ".join(cap_info["provides"][:5]) if cap_info["provides"] else "unknown"

            explanations.append(
                f"**{index}** ({cap_info['type']}): {cap_info['description']}. "
                f"Provides: {provides_text}."
            )

        return "\n".join(explanations)


# Global client instance
_splunk_client: Optional[SplunkClient] = None


def get_splunk_client() -> SplunkClient:
    """Get or create the global Splunk client instance."""
    global _splunk_client
    if _splunk_client is None:
        _splunk_client = SplunkClient()
    return _splunk_client


def initialize_splunk_client(config: Optional[SplunkConfig] = None) -> SplunkClient:
    """Initialize the global Splunk client with optional config."""
    global _splunk_client
    _splunk_client = SplunkClient(config)
    return _splunk_client
