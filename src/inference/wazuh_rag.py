"""
Wazuh RAG (Retrieval Augmented Generation) Module

Provides context-aware Wazuh field and query knowledge for improved SPL generation.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load sentence transformers to avoid slow startup
_model = None
_embeddings_cache = None


def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model for Wazuh RAG...")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            _model = None
    return _model


class WazuhRAG:
    """RAG system for Wazuh-specific SPL query generation."""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the Wazuh RAG system.

        Args:
            data_path: Path to wazuh_field_data.json. If None, uses default location.
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "wazuh_rag" / "wazuh_field_data.json"

        self.data_path = Path(data_path)
        self.data = None
        self.documents = []
        self.embeddings = None
        self.is_initialized = False

        # Pre-built Wazuh knowledge (field descriptions)
        self.field_descriptions = self._build_field_descriptions()

        # Example SPL queries for common Wazuh use cases
        self.example_queries = self._build_example_queries()

    def _build_field_descriptions(self) -> Dict[str, str]:
        """Build descriptions for common Wazuh fields."""
        return {
            # Agent fields
            "agent.id": "Unique identifier for the Wazuh agent (e.g., '001', '002')",
            "agent.name": "Hostname or name of the monitored endpoint",
            "agent.ip": "IP address of the Wazuh agent",
            "agent.labels.*": "Custom labels assigned to agents for grouping/filtering",

            # Rule fields
            "rule.id": "Unique rule identifier (e.g., '5501', '31168')",
            "rule.level": "Alert severity level (0-16, where 0=ignored, 15=severe attack)",
            "rule.description": "Human-readable description of what the rule detects",
            "rule.groups{}": "Categories the rule belongs to (e.g., 'authentication_failed', 'syscheck', 'web')",
            "rule.mitre.id{}": "MITRE ATT&CK technique IDs (e.g., 'T1110', 'T1059')",
            "rule.mitre.tactic{}": "MITRE ATT&CK tactics (e.g., 'Credential Access', 'Execution')",
            "rule.mitre.technique{}": "MITRE ATT&CK technique names",
            "rule.pci_dss{}": "PCI DSS compliance requirements matched",
            "rule.gdpr{}": "GDPR compliance requirements matched",
            "rule.hipaa{}": "HIPAA compliance requirements matched",
            "rule.nist_800_53{}": "NIST 800-53 controls matched",

            # Data fields (extracted from logs)
            "data.srcip": "Source IP address from the log event",
            "data.dstip": "Destination IP address",
            "data.srcport": "Source port number",
            "data.dstport": "Destination port number",
            "data.srcuser": "Source username (who performed the action)",
            "data.dstuser": "Destination/target username",
            "data.protocol": "Network protocol (TCP, UDP, etc.)",
            "data.url": "URL accessed or requested",
            "data.command": "Command executed (for command execution alerts)",
            "data.uid": "User ID from the system",
            "data.tty": "Terminal/TTY used",
            "data.pwd": "Working directory",

            # Vulnerability fields
            "data.vulnerability.cve": "CVE identifier for vulnerabilities",
            "data.vulnerability.severity": "Vulnerability severity (Critical, High, Medium, Low)",
            "data.vulnerability.package.name": "Affected software package name",
            "data.vulnerability.package.version": "Affected package version",
            "data.vulnerability.cvss.cvss3.base_score": "CVSS v3 base score (0-10)",
            "data.vulnerability.status": "Vulnerability status (e.g., 'Active', 'Solved')",

            # Check/SCA fields (Security Configuration Assessment)
            "check.id": "Security check identifier",
            "check.title": "Security check title/name",
            "check.result": "Check result (passed/failed)",
            "check.compliance.*": "Compliance framework mappings (CIS, PCI, etc.)",

            # Decoder/Location
            "decoder.name": "Decoder used to parse the log",
            "location": "Log source path or identifier",
            "manager.name": "Wazuh manager server name",
            "timestamp": "Event timestamp",
            "full_log": "Complete original log message",
        }

    def _build_example_queries(self) -> List[Dict[str, str]]:
        """Build example SPL queries for common Wazuh use cases."""
        return [
            {
                "description": "Find failed authentication attempts",
                "query": 'index=wazuh-alerts rule.groups{}="authentication_failed" | stats count by agent.name, data.srcip, data.srcuser',
                "use_case": "authentication failures, failed logins, brute force detection"
            },
            {
                "description": "Find successful authentications from specific IP",
                "query": 'index=wazuh-alerts rule.groups{}="authentication_success" data.srcip="10.0.0.1" | table timestamp, agent.name, data.srcuser',
                "use_case": "successful logins, authentication tracking"
            },
            {
                "description": "Find high severity alerts (level 10+)",
                "query": 'index=wazuh-alerts rule.level>=10 | stats count by rule.id, rule.description, rule.level | sort -rule.level',
                "use_case": "high severity alerts, critical events, security incidents"
            },
            {
                "description": "Find alerts by MITRE ATT&CK technique",
                "query": 'index=wazuh-alerts rule.mitre.id{}="T1110" | stats count by agent.name, rule.description',
                "use_case": "MITRE ATT&CK, brute force (T1110), specific technique detection"
            },
            {
                "description": "Find vulnerability alerts by severity",
                "query": 'index=wazuh-alerts data.vulnerability.severity="Critical" | stats count by agent.name, data.vulnerability.cve, data.vulnerability.package.name',
                "use_case": "vulnerabilities, CVE, critical vulnerabilities, patch management"
            },
            {
                "description": "Find file integrity monitoring (FIM) alerts",
                "query": 'index=wazuh-alerts rule.groups{}="syscheck" | stats count by agent.name, rule.description, syscheck.path',
                "use_case": "file integrity, FIM, syscheck, file changes"
            },
            {
                "description": "Find security configuration assessment failures",
                "query": 'index=wazuh-alerts check.result="failed" | stats count by agent.name, check.title, check.compliance.cis',
                "use_case": "SCA, security configuration, compliance, CIS benchmarks"
            },
            {
                "description": "Find command execution alerts",
                "query": 'index=wazuh-alerts data.command=* | table timestamp, agent.name, data.srcuser, data.command, data.pwd',
                "use_case": "command execution, shell commands, user activity"
            },
            {
                "description": "Find network connection alerts",
                "query": 'index=wazuh-alerts data.srcip=* data.dstip=* | stats count by data.srcip, data.dstip, data.dstport, data.protocol',
                "use_case": "network connections, firewall, network traffic"
            },
            {
                "description": "Find alerts for specific agent",
                "query": 'index=wazuh-alerts agent.name="server01" | timechart count by rule.groups{}',
                "use_case": "agent-specific, endpoint monitoring, specific host"
            },
            {
                "description": "Find web attack attempts",
                "query": 'index=wazuh-alerts rule.groups{}="web" rule.groups{}="attack" | stats count by rule.id, rule.description, data.srcip',
                "use_case": "web attacks, web application security, HTTP attacks"
            },
            {
                "description": "Find rootkit detection alerts",
                "query": 'index=wazuh-alerts rule.groups{}="rootcheck" | stats count by agent.name, rule.description',
                "use_case": "rootkit, malware, rootcheck"
            },
            {
                "description": "Count alerts by agent over time",
                "query": 'index=wazuh-alerts | timechart span=1h count by agent.name',
                "use_case": "alert trends, timeline, activity over time"
            },
            {
                "description": "Find PCI DSS compliance violations",
                "query": 'index=wazuh-alerts rule.pci_dss{}=* rule.level>=7 | stats count by rule.pci_dss{}, rule.description',
                "use_case": "PCI DSS, compliance, payment card industry"
            },
            {
                "description": "Find privilege escalation attempts",
                "query": 'index=wazuh-alerts rule.mitre.tactic{}="Privilege Escalation" | stats count by agent.name, rule.description, data.srcuser',
                "use_case": "privilege escalation, sudo, admin access"
            },
        ]

    def initialize(self) -> bool:
        """
        Initialize the RAG system by loading data and building embeddings.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self.is_initialized:
            return True

        try:
            # Load Wazuh field data from Splunk export
            if self.data_path.exists():
                with open(self.data_path) as f:
                    self.data = json.load(f)
                logger.info(f"Loaded Wazuh data from {self.data_path}")
            else:
                logger.warning(f"Wazuh data file not found at {self.data_path}, using built-in knowledge only")
                self.data = {}

            # Build document corpus for embedding
            self._build_documents()

            # Build embeddings
            self._build_embeddings()

            self.is_initialized = True
            logger.info(f"Wazuh RAG initialized with {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Wazuh RAG: {e}")
            return False

    def _build_documents(self):
        """Build the document corpus for semantic search."""
        self.documents = []

        # Add field descriptions
        for field, description in self.field_descriptions.items():
            self.documents.append({
                "type": "field",
                "content": f"Field: {field} - {description}",
                "field": field,
                "description": description
            })

        # Add example queries
        for example in self.example_queries:
            self.documents.append({
                "type": "example_query",
                "content": f"Use case: {example['use_case']}. Query: {example['query']}. Description: {example['description']}",
                "query": example["query"],
                "description": example["description"],
                "use_case": example["use_case"]
            })

        # Add rule information from Splunk data
        if self.data and "rule_ids" in self.data:
            for rule in self.data["rule_ids"][:100]:  # Top 100 rules
                rule_id = rule.get("rule.id", "")
                desc = rule.get("rule.description", "")
                level = rule.get("rule.level", "")
                count = rule.get("count", "")
                if rule_id and desc:
                    self.documents.append({
                        "type": "rule",
                        "content": f"Wazuh Rule {rule_id} (level {level}): {desc}. Seen {count} times.",
                        "rule_id": rule_id,
                        "description": desc,
                        "level": level
                    })

        # Add MITRE mappings from Splunk data
        if self.data and "mitre_mappings" in self.data:
            for mitre in self.data["mitre_mappings"]:
                mitre_id = mitre.get("rule.mitre.id{}", "")
                technique = mitre.get("rule.mitre.technique{}", "")
                tactic = mitre.get("rule.mitre.tactic{}", "")
                count = mitre.get("count", "")
                if mitre_id:
                    self.documents.append({
                        "type": "mitre",
                        "content": f"MITRE ATT&CK {mitre_id}: {technique} ({tactic}). Seen {count} times in your environment.",
                        "mitre_id": mitre_id,
                        "technique": technique,
                        "tactic": tactic
                    })

        # Add rule groups from Splunk data
        if self.data and "rule_groups" in self.data:
            for group in self.data["rule_groups"]:
                group_name = group.get("rule.groups{}", "")
                count = group.get("count", "")
                if group_name:
                    self.documents.append({
                        "type": "rule_group",
                        "content": f"Rule group '{group_name}' - use rule.groups{{}}=\"{group_name}\" to filter. {count} events.",
                        "group": group_name,
                        "count": count
                    })

        # Add actual fields from Splunk data
        if self.data and "fields" in self.data:
            core_fields = self.data["fields"].get("core", [])
            for field in core_fields:
                field_name = field.get("field", "")
                count = field.get("count", "")
                distinct = field.get("distinct_count", "")
                if field_name and field_name not in self.field_descriptions:
                    self.documents.append({
                        "type": "field_dynamic",
                        "content": f"Field: {field_name} - {count} occurrences, {distinct} unique values",
                        "field": field_name,
                        "count": count,
                        "distinct_count": distinct
                    })

    def _build_embeddings(self):
        """Build embeddings for all documents."""
        model = get_embedding_model()
        if model is None:
            logger.warning("No embedding model available, falling back to keyword matching")
            self.embeddings = None
            return

        try:
            texts = [doc["content"] for doc in self.documents]
            self.embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            logger.info(f"Built embeddings for {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to build embeddings: {e}")
            self.embeddings = None

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for relevant context based on the query.

        Args:
            query: User's natural language query
            top_k: Number of results to return

        Returns:
            List of relevant documents
        """
        if not self.is_initialized:
            self.initialize()

        if not self.documents:
            return []

        model = get_embedding_model()

        # Try semantic search if embeddings available
        if model is not None and self.embeddings is not None:
            try:
                query_embedding = model.encode([query], convert_to_numpy=True)[0]
                similarities = np.dot(self.embeddings, query_embedding)
                top_indices = np.argsort(similarities)[-top_k:][::-1]

                results = []
                for idx in top_indices:
                    doc = self.documents[idx].copy()
                    doc["similarity"] = float(similarities[idx])
                    results.append(doc)
                return results
            except Exception as e:
                logger.warning(f"Semantic search failed, falling back to keyword: {e}")

        # Fallback to keyword matching
        query_lower = query.lower()
        scored_docs = []
        for doc in self.documents:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            if score > 0:
                doc_copy = doc.copy()
                doc_copy["similarity"] = score
                scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_docs[:top_k]

    def get_context_for_query(self, user_query: str) -> str:
        """
        Get formatted context string to add to the system prompt.

        Args:
            user_query: User's natural language query

        Returns:
            Formatted context string
        """
        # ALWAYS start with the strict field reference
        context_parts = [self._get_strict_wazuh_prompt()]

        results = self.search(user_query, top_k=8)

        if not results:
            return "\n".join(context_parts)

        context_parts.append("\n## Additional Relevant Context\n")

        # Group results by type
        fields = [r for r in results if r["type"] in ("field", "field_dynamic")]
        examples = [r for r in results if r["type"] == "example_query"]
        rules = [r for r in results if r["type"] == "rule"]
        mitre = [r for r in results if r["type"] == "mitre"]
        groups = [r for r in results if r["type"] == "rule_group"]

        # Add relevant fields
        if fields:
            context_parts.append("### Relevant Fields:")
            for f in fields[:5]:
                context_parts.append(f"- {f.get('field', '')}: {f.get('description', f.get('content', ''))}")

        # Add example queries
        if examples:
            context_parts.append("\n### Similar Query Examples:")
            for e in examples[:3]:
                context_parts.append(f"- {e.get('description', '')}")
                context_parts.append(f"  ```{e.get('query', '')}```")

        # Add rule info
        if rules:
            context_parts.append("\n### Relevant Rules:")
            for r in rules[:3]:
                context_parts.append(f"- Rule {r.get('rule_id', '')}: {r.get('description', '')}")

        # Add MITRE info
        if mitre:
            context_parts.append("\n### MITRE ATT&CK Mappings:")
            for m in mitre[:3]:
                context_parts.append(f"- {m.get('mitre_id', '')}: {m.get('technique', '')} ({m.get('tactic', '')})")

        # Add rule groups
        if groups:
            context_parts.append("\n### Rule Groups to Filter:")
            for g in groups[:3]:
                context_parts.append(f"- rule.groups{{}}=\"{g.get('group', '')}\"")

        return "\n".join(context_parts)

    def _get_default_context(self) -> str:
        """Get default Wazuh context when no specific matches found."""
        return self._get_strict_wazuh_prompt()

    def _get_strict_wazuh_prompt(self) -> str:
        """Get strict Wazuh prompt with exact field names."""
        return """
=== WAZUH-ALERTS QUERY GENERATION RULES ===

YOU ARE GENERATING A QUERY FOR THE WAZUH-ALERTS INDEX. THIS IS A WAZUH EDR/SECURITY LOG INDEX.

MANDATORY RULES - FOLLOW EXACTLY:

1. INDEX: ALWAYS use index=wazuh-alerts (NEVER use wazuh-logs, wazuh-sysmon, or other indexes)

2. FIELD NAME TRANSLATIONS - USE THE RIGHT COLUMN, NEVER THE LEFT:
   WRONG              → CORRECT
   src_ip             → data.srcip
   dest_ip            → data.dstip
   dst_ip             → data.dstip
   src_port           → data.srcport
   dest_port          → data.dstport
   dst_port           → data.dstport
   user               → data.srcuser
   src_user           → data.srcuser
   dest_user          → data.dstuser
   status_code        → (NO EQUIVALENT - use rule.groups{} or data.protocol)
   http_status_code   → (NO EQUIVALENT - filter by rule.groups{}="web")
   http_method        → data.protocol
   action             → (NO EQUIVALENT - use rule.groups{} for filtering)
   event_type         → (NO EQUIVALENT - use rule.groups{} or rule.id)
   EventCode          → (NO EQUIVALENT - use rule.id)
   method             → data.protocol
   id.orig_h          → data.srcip
   alert.signature    → rule.description
   severity           → rule.level
   host               → agent.name

3. SOURCETYPE: ONLY use sourcetype=wazuh-alerts or omit it entirely

4. AVAILABLE FIELDS IN WAZUH-ALERTS:
   - agent.id, agent.name, agent.ip (agent info)
   - rule.id, rule.level, rule.description, rule.groups{}, rule.mitre.id{}, rule.mitre.tactic{} (rule info)
   - data.srcip, data.dstip, data.srcport, data.dstport (network)
   - data.srcuser, data.dstuser (users)
   - data.protocol, data.url, data.command, data.pwd, data.tty, data.uid (activity)
   - timestamp, location, decoder.name, full_log, manager.name (metadata)

   IMPORTANT: rule.level is a MULTIVALUE field stored as strings.
   For filtering by severity level, use IN operator:
   rule.level IN (7, 8, 9, 10, 11, 12, 13, 14, 15)  -- for level 7+
   rule.level IN (10, 11, 12, 13, 14, 15)  -- for level 10+
   NOT: rule.level>=10 or tonumber(rule.level)>=10 (these don't work!)

5. RULE GROUPS FOR FILTERING (use rule.groups{}="value"):
   - "web" - web traffic/HTTP logs
   - "attack" - attack detection
   - "authentication_failed" - failed logins
   - "authentication_success" - successful logins
   - "syscheck" - file integrity monitoring
   - "vulnerability-detector" - vulnerability alerts
   - "sudo" - sudo commands
   - "pam" - PAM authentication

EXAMPLE QUERIES (USE THESE PATTERNS):

Web/HTTP traffic:
index=wazuh-alerts rule.groups{}="web" | stats count by data.srcip, data.url, data.protocol

Failed authentication:
index=wazuh-alerts rule.groups{}="authentication_failed" | stats count by agent.name, data.srcip

High severity alerts (level 7+):
index=wazuh-alerts rule.level IN (7, 8, 9, 10, 11, 12, 13, 14, 15) | stats count by rule.id, rule.description

MITRE ATT&CK technique:
index=wazuh-alerts rule.mitre.id{}="T1110" | stats count by agent.name, data.srcip

Network connections:
index=wazuh-alerts data.srcip=* data.dstip=* | stats count by data.srcip, data.dstip, data.dstport

Sudo commands:
index=wazuh-alerts rule.groups{}="sudo" | stats count by agent.name, data.srcuser, data.command

Successful authentication:
index=wazuh-alerts rule.groups{}="authentication_success" | table timestamp, agent.name, data.srcuser

List rule IDs (simple):
index=wazuh-alerts | stats count by rule.id, rule.description | sort -count

Count events by agent:
index=wazuh-alerts | stats count by agent.name | sort -count

KEEP QUERIES SIMPLE - avoid complex eval, while loops, or nested functions

=== END WAZUH RULES - GENERATE QUERY NOW ===
"""


# Singleton instance
_wazuh_rag_instance = None


def get_wazuh_rag() -> WazuhRAG:
    """Get the singleton WazuhRAG instance."""
    global _wazuh_rag_instance
    if _wazuh_rag_instance is None:
        _wazuh_rag_instance = WazuhRAG()
    return _wazuh_rag_instance


def get_wazuh_context(user_query: str, indexes: Optional[List[str]] = None) -> Optional[str]:
    """
    Get Wazuh-specific context if the query is targeting a Wazuh index.

    Args:
        user_query: User's natural language query
        indexes: List of indexes being queried

    Returns:
        Context string if Wazuh index, None otherwise
    """
    # Check if querying Wazuh index
    if indexes:
        wazuh_indexes = [idx for idx in indexes if "wazuh" in idx.lower()]
        if not wazuh_indexes:
            return None

    rag = get_wazuh_rag()
    if not rag.is_initialized:
        rag.initialize()

    return rag.get_context_for_query(user_query)


def fix_wazuh_query(query: str, indexes: Optional[List[str]] = None) -> str:
    """
    Post-process a generated query to fix common Wazuh field name errors.

    Args:
        query: Generated SPL query
        indexes: List of indexes being queried

    Returns:
        Fixed query with correct Wazuh field names
    """
    # Only fix if targeting Wazuh index
    if indexes:
        wazuh_indexes = [idx for idx in indexes if "wazuh" in idx.lower()]
        if not wazuh_indexes:
            return query
    elif "wazuh" not in query.lower():
        return query

    import re

    fixed = query

    # First, remove problematic subsearches entirely
    fixed = re.sub(r'\[search\s+index=wazuh-logs[^\]]*\]', '', fixed, flags=re.IGNORECASE)
    fixed = re.sub(r'\[search\s+index=wazuh-sysmon[^\]]*\]', '', fixed, flags=re.IGNORECASE)
    fixed = re.sub(r'\[search\s+index=wazuh_logs[^\]]*\]', '', fixed, flags=re.IGNORECASE)

    # Remove XmlWinEventLog sourcetypes entirely
    fixed = re.sub(r'sourcetype="XmlWinEventLog:[^"]*"\s*', '', fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"sourcetype='XmlWinEventLog:[^']*'\s*", '', fixed, flags=re.IGNORECASE)

    # Field name replacements (order matters - more specific first)
    replacements = [
        # Wrong indexes/sourcetypes
        (r'index=wazuh-logs', 'index=wazuh-alerts'),
        (r'index=wazuh-sysmon', 'index=wazuh-alerts'),
        (r'index=wazuh_logs', 'index=wazuh-alerts'),
        (r'index=wazuh_sysmon', 'index=wazuh-alerts'),
        (r'sourcetype=wazuh-sysmon', 'sourcetype=wazuh-alerts'),
        (r'sourcetype=wazuh_sysmon', 'sourcetype=wazuh-alerts'),
        (r'sourcetype=wazuh-edr', 'sourcetype=wazuh-alerts'),
        (r'sourcetype=wazuh_edr', 'sourcetype=wazuh-alerts'),
        (r'sourcetype=access_combined', 'sourcetype=wazuh-alerts'),
        (r'sourcetype=rdp_logs', 'sourcetype=wazuh-alerts'),

        # Remove data. prefix from wrong fields first (to avoid data.data.)
        (r'data\.http_status_code', 'rule.groups{}'),
        (r'data\.http_method', 'data.protocol'),
        (r'data\.status_code', 'rule.groups{}'),

        # Network fields - be careful with word boundaries, but NOT after data.
        (r'(?<!data\.)\bsrc_ip\b', 'data.srcip'),
        (r'(?<!data\.)\bdest_ip\b', 'data.dstip'),
        (r'(?<!data\.)\bdst_ip\b', 'data.dstip'),
        (r'(?<!data\.)\bsrc_port\b', 'data.srcport'),
        (r'(?<!data\.)\bdest_port\b', 'data.dstport'),
        (r'(?<!data\.)\bdst_port\b', 'data.dstport'),
        (r'\bid\.orig_h\b', 'data.srcip'),
        (r'\bid\.resp_h\b', 'data.dstip'),

        # User fields - not after data.
        (r'(?<!data\.)\bsrc_user\b', 'data.srcuser'),
        (r'(?<!data\.)\bdest_user\b', 'data.dstuser'),
        # Only replace standalone 'user' (not in user.agent, data.srcuser, etc.)
        (r'(?<![a-zA-Z_.])user(?=[,\s|=]|$)', 'data.srcuser'),

        # HTTP/Web fields - only if not already prefixed
        (r'(?<!data\.)\bhttp_status_code\b', 'rule.groups{}'),
        (r'(?<!data\.)\bstatus_code\b', 'rule.groups{}'),
        (r'(?<!data\.)\bhttp_method\b', 'data.protocol'),

        # Alert fields
        (r'\balert\.signature\b', 'rule.description'),
        (r'\balert\.severity\b', 'rule.level'),
        (r'(?<!rule\.)\bseverity\b(?=[=\s|,])', 'rule.level'),

        # Host field - only standalone
        (r'(?<![a-zA-Z_.])\bhost\b(?=[,\s|=])', 'agent.name'),

        # Event fields that don't exist in Wazuh
        (r'\bEventCode\b', 'rule.id'),
        (r'(?<!rule\.)\bevent_type\b', 'rule.groups{}'),
        (r'(?<!rule\.)\beventType\b', 'rule.groups{}'),

        # Action field doesn't exist - remove or replace with rule.groups{}
        (r'\baction=blocked\b', 'rule.groups{}="access_denied"'),
        (r'\baction=allowed\b', 'rule.groups{}="authentication_success"'),
        (r'\baction=[^\s|]+\s*', ''),  # remove other action= patterns
    ]

    # Special fix: rule.level is multivalue, need IN operator for comparisons
    # Convert rule.level>=X or rule.level>X to rule.level IN (X, X+1, ...)
    def make_level_in_clause(match):
        op = match.group(1)
        val = int(match.group(2))
        if '>=' in op:
            levels = list(range(val, 16))  # Wazuh levels go up to 15
        elif '>' in op:
            levels = list(range(val + 1, 16))
        elif '<=' in op:
            levels = list(range(0, val + 1))
        elif '<' in op:
            levels = list(range(0, val))
        else:
            return match.group(0)  # Return unchanged for = or !=
        return f"rule.level IN ({', '.join(str(l) for l in levels)})"

    # Fix rule.level comparisons in search clause
    fixed = re.sub(
        r'rule\.level\s*([><=]+)\s*(\d+)',
        make_level_in_clause,
        fixed,
        flags=re.IGNORECASE
    )

    # Also fix any tonumber(rule.level) patterns (from previous runs)
    fixed = re.sub(
        r'\|\s*where\s+tonumber\(rule\.level\)\s*([><=]+)\s*(\d+)',
        lambda m: f'rule.level IN ({", ".join(str(l) for l in range(int(m.group(2)) if ">=" in m.group(1) else int(m.group(2))+1, 16))})',
        fixed,
        flags=re.IGNORECASE
    )

    for pattern, replacement in replacements:
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

    # Clean up any double spaces or empty pipes
    fixed = re.sub(r'\s+', ' ', fixed)
    fixed = re.sub(r'\|\s*\|', '|', fixed)
    fixed = re.sub(r'\s+\|', ' |', fixed)
    fixed = re.sub(r'\|\s+', '| ', fixed)

    # Ensure index=wazuh-alerts is present at the start
    if not re.search(r'index=wazuh-alerts', fixed, re.IGNORECASE):
        fixed = 'index=wazuh-alerts ' + fixed

    logger.info(f"Fixed Wazuh query: {query[:100]}... -> {fixed[:100]}...")

    return fixed.strip()
