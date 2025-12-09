"""
SPL RAG (Retrieval Augmented Generation) Module for Educational Queries

Provides knowledge about Splunk Processing Language (SPL) commands, functions,
and best practices to help analysts learn SPL.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load sentence transformers to avoid slow startup
_model = None


def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model for SPL RAG...")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            _model = None
    return _model


class SPLRAG:
    """RAG system for SPL educational content."""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the SPL RAG system.

        Args:
            data_path: Path to spl_knowledge.json. If None, uses default location.
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "spl_knowledge" / "spl_knowledge.json"

        self.data_path = Path(data_path)
        self.data = None
        self.documents = []
        self.embeddings = None
        self.is_initialized = False

        # Built-in SPL knowledge
        self.commands = self._build_command_knowledge()
        self.functions = self._build_function_knowledge()
        self.concepts = self._build_concept_knowledge()
        self.examples = self._build_example_knowledge()
        self.best_practices = self._build_best_practices()

    def _build_command_knowledge(self) -> Dict[str, Dict]:
        """Build knowledge base for SPL commands."""
        return {
            # Search Commands
            "search": {
                "name": "search",
                "category": "search",
                "syntax": "search <search-expression>",
                "description": "The search command is the foundation of every SPL query. It retrieves events from indexes that match your criteria. Usually implicit at the start of a query.",
                "examples": [
                    'search index=main error',
                    'search index=security sourcetype=access_combined status>=400',
                    'index=* "connection refused"'
                ],
                "tips": [
                    "The 'search' keyword is optional at the start of a query",
                    "Use wildcards (*) sparingly as they can slow down searches",
                    "Always specify an index when possible for better performance",
                    "Use quotes around phrases with spaces"
                ],
                "related": ["where", "filter"]
            },
            "where": {
                "name": "where",
                "category": "filtering",
                "syntax": "where <eval-expression>",
                "description": "Filters results using eval expressions. More powerful than search for complex conditions because it supports functions and calculations.",
                "examples": [
                    '| where status > 400',
                    '| where like(user, "admin%")',
                    '| where len(message) > 100',
                    '| where isnotnull(src_ip) AND isnotnull(dest_ip)'
                ],
                "tips": [
                    "Use 'where' after initial search for complex filtering",
                    "Supports all eval functions",
                    "Field names are case-sensitive",
                    "Use like() for pattern matching instead of regex when possible"
                ],
                "related": ["search", "eval", "filter"]
            },
            "stats": {
                "name": "stats",
                "category": "aggregation",
                "syntax": "stats <stats-function>... [by <field-list>]",
                "description": "Calculates aggregate statistics over events. One of the most commonly used commands for summarizing data.",
                "examples": [
                    '| stats count',
                    '| stats count by host',
                    '| stats avg(response_time) min(response_time) max(response_time) by endpoint',
                    '| stats sum(bytes) as total_bytes by src_ip | sort -total_bytes',
                    '| stats dc(user) as unique_users by department',
                    '| stats values(action) as actions by user'
                ],
                "tips": [
                    "Use 'as' to rename output fields",
                    "dc() counts distinct/unique values",
                    "values() lists all unique values",
                    "list() preserves all values including duplicates",
                    "earliest() and latest() get time boundaries"
                ],
                "functions": ["count", "sum", "avg", "min", "max", "dc", "values", "list", "first", "last", "earliest", "latest", "stdev", "var", "median", "mode", "perc"],
                "related": ["eventstats", "streamstats", "chart", "timechart"]
            },
            "eval": {
                "name": "eval",
                "category": "transformation",
                "syntax": "eval <field>=<expression>",
                "description": "Creates or modifies fields using expressions. Essential for data transformation, calculations, and conditional logic.",
                "examples": [
                    '| eval response_sec = response_time/1000',
                    '| eval status_category = if(status<400, "success", "error")',
                    '| eval full_name = first_name . " " . last_name',
                    '| eval is_weekend = if(strftime(_time, "%u") > 5, "yes", "no")',
                    '| eval severity = case(level<4, "low", level<7, "medium", level<10, "high", true(), "critical")'
                ],
                "tips": [
                    "Use 'case()' for multiple conditions (like switch statement)",
                    "Use 'if()' for simple true/false conditions",
                    "String concatenation uses the dot (.) operator",
                    "Use 'coalesce()' to handle null values",
                    "Use 'mvindex()' to access multivalue field elements"
                ],
                "functions": ["if", "case", "coalesce", "null", "nullif", "tonumber", "tostring", "len", "lower", "upper", "substr", "replace", "split", "mvcount", "mvindex", "mvjoin"],
                "related": ["where", "rex", "rename"]
            },
            "table": {
                "name": "table",
                "category": "display",
                "syntax": "table <field-list>",
                "description": "Displays specified fields in a table format. Use at the end of your query to show only the fields you need.",
                "examples": [
                    '| table _time host source message',
                    '| table user src_ip action timestamp',
                    '| table _time, user, action, status'
                ],
                "tips": [
                    "Fields are displayed in the order specified",
                    "Unlisted fields are hidden from output",
                    "Use with 'rename' to improve column headers",
                    "Commas between fields are optional"
                ],
                "related": ["fields", "rename", "format"]
            },
            "fields": {
                "name": "fields",
                "category": "display",
                "syntax": "fields [+|-] <field-list>",
                "description": "Keeps or removes fields from results. Use + to keep (default), - to remove.",
                "examples": [
                    '| fields host source message',
                    '| fields - _raw _time',
                    '| fields + user action status'
                ],
                "tips": [
                    "Use 'fields' early in pipeline to improve performance",
                    "Removing _raw can significantly speed up searches",
                    "'fields' differs from 'table' - table also formats output"
                ],
                "related": ["table", "rename"]
            },
            "rename": {
                "name": "rename",
                "category": "transformation",
                "syntax": "rename <field> as <newfield>",
                "description": "Renames fields in your results. Useful for making output more readable.",
                "examples": [
                    '| rename src_ip as "Source IP"',
                    '| rename count as "Event Count", user as "Username"',
                    '| rename _time as timestamp'
                ],
                "tips": [
                    "Use quotes for field names with spaces",
                    "Can rename multiple fields in one command",
                    "Renamed fields must be referenced by new name downstream"
                ],
                "related": ["table", "fields", "eval"]
            },
            "sort": {
                "name": "sort",
                "category": "ordering",
                "syntax": "sort [+|-] <field> [limit=<int>]",
                "description": "Orders results by specified fields. Use - for descending, + for ascending (default).",
                "examples": [
                    '| sort -count',
                    '| sort _time',
                    '| sort -count limit=10',
                    '| sort -severity +host'
                ],
                "tips": [
                    "Minus (-) means descending (highest first)",
                    "Plus (+) means ascending (lowest first)",
                    "Use limit to get top/bottom N results",
                    "Sorting is case-sensitive for strings"
                ],
                "related": ["head", "tail", "top", "rare"]
            },
            "head": {
                "name": "head",
                "category": "limiting",
                "syntax": "head [<N>]",
                "description": "Returns the first N results (default 10). Events are returned in the order they were processed.",
                "examples": [
                    '| head',
                    '| head 5',
                    '| head 100'
                ],
                "tips": [
                    "Default is 10 results",
                    "Use after 'sort' to get top N by a field",
                    "More efficient than 'sort limit=N' for large datasets"
                ],
                "related": ["tail", "sort", "top"]
            },
            "tail": {
                "name": "tail",
                "category": "limiting",
                "syntax": "tail [<N>]",
                "description": "Returns the last N results (default 10).",
                "examples": [
                    '| tail',
                    '| tail 20'
                ],
                "tips": [
                    "Returns results in reverse order",
                    "Useful for getting oldest events when sorted by time"
                ],
                "related": ["head", "sort"]
            },
            "top": {
                "name": "top",
                "category": "aggregation",
                "syntax": "top [<N>] <field> [by <field>]",
                "description": "Returns the most common values of a field. Automatically calculates count and percent.",
                "examples": [
                    '| top user',
                    '| top 10 src_ip',
                    '| top status by host',
                    '| top 5 error_code showperc=false'
                ],
                "tips": [
                    "Default N is 10",
                    "Use showperc=false to hide percentage",
                    "Use countfield= to rename the count column",
                    "Shortcut for 'stats count by field | sort -count | head N'"
                ],
                "related": ["rare", "stats", "sort"]
            },
            "rare": {
                "name": "rare",
                "category": "aggregation",
                "syntax": "rare [<N>] <field> [by <field>]",
                "description": "Returns the least common values of a field. Opposite of 'top'.",
                "examples": [
                    '| rare user',
                    '| rare 10 error_code',
                    '| rare status by host'
                ],
                "tips": [
                    "Useful for finding anomalies",
                    "Same options as 'top' command"
                ],
                "related": ["top", "stats"]
            },
            "timechart": {
                "name": "timechart",
                "category": "visualization",
                "syntax": "timechart [span=<time>] <stats-function> [by <field>]",
                "description": "Creates time-series charts by aggregating over time buckets. Essential for trend analysis.",
                "examples": [
                    '| timechart count',
                    '| timechart span=1h count by status',
                    '| timechart span=5m avg(response_time)',
                    '| timechart span=1d sum(bytes) by host limit=5'
                ],
                "tips": [
                    "span= controls time bucket size (1m, 5m, 1h, 1d, etc.)",
                    "limit= restricts number of series when using 'by'",
                    "Use 'useother=false' to hide the 'OTHER' series",
                    "Great for visualizing trends over time"
                ],
                "related": ["chart", "stats", "bucket"]
            },
            "chart": {
                "name": "chart",
                "category": "visualization",
                "syntax": "chart <stats-function> [over <field>] [by <field>]",
                "description": "Creates charts by aggregating over field values (not time). Use for non-time-based visualizations.",
                "examples": [
                    '| chart count over status',
                    '| chart count over status by host',
                    '| chart avg(response_time) over endpoint by method'
                ],
                "tips": [
                    "'over' is the x-axis",
                    "'by' creates separate series",
                    "Use 'timechart' if x-axis should be time"
                ],
                "related": ["timechart", "stats"]
            },
            "dedup": {
                "name": "dedup",
                "category": "filtering",
                "syntax": "dedup [<N>] <field-list>",
                "description": "Removes duplicate events based on specified fields. Keeps the first N occurrences.",
                "examples": [
                    '| dedup user',
                    '| dedup src_ip dest_ip',
                    '| dedup 3 host',
                    '| dedup user sortby -_time'
                ],
                "tips": [
                    "Default keeps first occurrence",
                    "Use sortby= to control which duplicate is kept",
                    "Useful for getting unique combinations"
                ],
                "related": ["uniq", "stats dc"]
            },
            "rex": {
                "name": "rex",
                "category": "extraction",
                "syntax": 'rex field=<field> "<regex>"',
                "description": "Extracts fields using regular expressions. Named capture groups become fields.",
                "examples": [
                    '| rex field=_raw "user=(?<username>\\w+)"',
                    '| rex field=message "error code: (?<error_code>\\d+)"',
                    '| rex field=url "https?://(?<domain>[^/]+)"',
                    '| rex mode=sed field=message "s/password=\\w+/password=REDACTED/g"'
                ],
                "tips": [
                    "Use named capture groups: (?<fieldname>pattern)",
                    "Default field is _raw",
                    "mode=sed allows sed-style substitution",
                    "Use max_match=0 to extract all matches"
                ],
                "related": ["eval", "extract", "regex"]
            },
            "regex": {
                "name": "regex",
                "category": "filtering",
                "syntax": 'regex <field>=<regex> | regex <regex>',
                "description": "Filters events that match a regular expression. Keep events where field matches pattern.",
                "examples": [
                    '| regex _raw="error|warning|critical"',
                    '| regex email="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"',
                    '| regex _raw!="DEBUG"'
                ],
                "tips": [
                    "Use != for negative matching (exclude matches)",
                    "Less efficient than indexed searches - use sparingly",
                    "Consider 'where match()' for more complex logic"
                ],
                "related": ["rex", "where", "search"]
            },
            "transaction": {
                "name": "transaction",
                "category": "correlation",
                "syntax": "transaction <field-list> [maxspan=<time>] [startswith=<search>] [endswith=<search>]",
                "description": "Groups events into transactions based on common field values. Useful for correlating related events.",
                "examples": [
                    '| transaction session_id',
                    '| transaction user maxspan=30m',
                    '| transaction host startswith="login" endswith="logout"',
                    '| transaction session_id maxspan=1h maxpause=5m'
                ],
                "tips": [
                    "Creates duration field automatically",
                    "eventcount shows number of events in transaction",
                    "Use maxspan to limit transaction duration",
                    "Resource intensive - use sparingly on large datasets"
                ],
                "related": ["stats", "eventstats"]
            },
            "join": {
                "name": "join",
                "category": "correlation",
                "syntax": "join [type=inner|outer|left] <field-list> [subsearch]",
                "description": "Combines results from two searches based on common fields. Like SQL JOIN.",
                "examples": [
                    '| join user [search index=users | fields user, department]',
                    '| join type=left src_ip [search index=assets | fields ip as src_ip, hostname]',
                    '| join type=outer user, host [search index=baseline]'
                ],
                "tips": [
                    "Default is inner join",
                    "Left join keeps all results from main search",
                    "Subsearch results are limited (50K events by default)",
                    "Consider 'lookup' for static data enrichment"
                ],
                "related": ["lookup", "append", "appendcols"]
            },
            "lookup": {
                "name": "lookup",
                "category": "enrichment",
                "syntax": "lookup <lookup-name> <field> [OUTPUT <field>]",
                "description": "Enriches events with data from lookup tables. More efficient than join for static reference data.",
                "examples": [
                    '| lookup user_info user OUTPUT department full_name',
                    '| lookup geo_ip src_ip OUTPUT country city',
                    '| lookup asset_inventory ip as dest_ip OUTPUTNEW hostname owner'
                ],
                "tips": [
                    "Lookup tables are pre-configured CSV files",
                    "OUTPUT overwrites existing fields, OUTPUTNEW doesn't",
                    "Use 'inputlookup' to view lookup table contents",
                    "Very efficient for enrichment"
                ],
                "related": ["join", "inputlookup", "outputlookup"]
            },
            "eventstats": {
                "name": "eventstats",
                "category": "aggregation",
                "syntax": "eventstats <stats-function> [by <field-list>]",
                "description": "Like stats, but adds aggregate values to each event instead of collapsing them. Great for comparison to group averages.",
                "examples": [
                    '| eventstats avg(response_time) as avg_response',
                    '| eventstats count as total_by_user by user',
                    '| eventstats avg(bytes) as dept_avg by department | eval above_avg=if(bytes>dept_avg, "yes", "no")'
                ],
                "tips": [
                    "Events are not collapsed like with 'stats'",
                    "Useful for calculating percentages of total",
                    "Can compare individual events to group statistics"
                ],
                "related": ["stats", "streamstats"]
            },
            "streamstats": {
                "name": "streamstats",
                "category": "aggregation",
                "syntax": "streamstats <stats-function> [by <field-list>] [window=<N>]",
                "description": "Calculates running/cumulative statistics as events stream through. Useful for running totals and moving averages.",
                "examples": [
                    '| streamstats count as running_total',
                    '| streamstats avg(response_time) as moving_avg window=10',
                    '| streamstats sum(bytes) as cumulative_bytes by user',
                    '| streamstats current=false window=5 avg(value) as prev_avg'
                ],
                "tips": [
                    "window= limits calculation to last N events",
                    "current=false excludes current event from calculation",
                    "Events must be in desired order (use sort first)",
                    "Great for detecting anomalies from baseline"
                ],
                "related": ["stats", "eventstats"]
            },
            "iplocation": {
                "name": "iplocation",
                "category": "enrichment",
                "syntax": "iplocation <ip-field>",
                "description": "Adds geographic information (country, city, region, lat/lon) to IP addresses.",
                "examples": [
                    '| iplocation src_ip',
                    '| iplocation clientip | stats count by Country',
                    '| iplocation dest_ip | geom geo_countries featureIdField=Country'
                ],
                "tips": [
                    "Creates: City, Country, Region, lat, lon fields",
                    "Works with both IPv4 and IPv6",
                    "Uses MaxMind GeoIP database"
                ],
                "related": ["lookup", "geom"]
            },
            "bucket": {
                "name": "bucket",
                "category": "transformation",
                "syntax": "bucket <field> [span=<value>] [bins=<N>]",
                "description": "Groups continuous values into discrete buckets. Useful for histograms and grouping numeric ranges.",
                "examples": [
                    '| bucket _time span=1h',
                    '| bucket response_time span=100',
                    '| bucket bytes bins=10',
                    '| bucket _time span=1d | stats count by _time'
                ],
                "tips": [
                    "span= sets bucket size (time or numeric)",
                    "bins= sets number of buckets",
                    "Also known as 'bin' command (alias)"
                ],
                "related": ["timechart", "chart"]
            },
            "fillnull": {
                "name": "fillnull",
                "category": "transformation",
                "syntax": "fillnull [value=<string>] [<field-list>]",
                "description": "Replaces null values with a specified value. Default replacement is 0.",
                "examples": [
                    '| fillnull',
                    '| fillnull value="N/A"',
                    '| fillnull value=0 count',
                    '| fillnull value="unknown" user department'
                ],
                "tips": [
                    "Default value is 0",
                    "Specify fields to only fill those",
                    "Useful before calculations to avoid nulls"
                ],
                "related": ["eval coalesce", "eval null"]
            },
            "append": {
                "name": "append",
                "category": "combining",
                "syntax": "append [subsearch]",
                "description": "Appends results from a subsearch to the current results. Like UNION in SQL.",
                "examples": [
                    '| append [search index=other_index | stats count]',
                    '| append [search index=security earliest=-7d | stats count as last_week]'
                ],
                "tips": [
                    "Results are added to the end",
                    "Fields don't need to match",
                    "Use appendcols to add columns instead of rows"
                ],
                "related": ["appendcols", "join", "union"]
            },
            "appendcols": {
                "name": "appendcols",
                "category": "combining",
                "syntax": "appendcols [subsearch]",
                "description": "Appends fields from a subsearch as new columns. Matches by row position.",
                "examples": [
                    '| appendcols [search index=baseline | stats avg(value) as baseline_avg]',
                    '| appendcols [stats count as total]'
                ],
                "tips": [
                    "Matches results by row position, not by field values",
                    "Subsearch should return same number of rows",
                    "Use join for matching by field values"
                ],
                "related": ["append", "join"]
            },
            "makemv": {
                "name": "makemv",
                "category": "transformation",
                "syntax": "makemv [delim=<string>] <field>",
                "description": "Converts a single-value field into a multivalue field by splitting on a delimiter.",
                "examples": [
                    '| makemv delim="," tags',
                    '| makemv delim=" " words',
                    '| makemv tokenizer="([^,]+)" items'
                ],
                "tips": [
                    "Default delimiter is single space",
                    "Use tokenizer= for regex-based splitting",
                    "Opposite of mvjoin (eval function)"
                ],
                "related": ["mvexpand", "eval mvjoin"]
            },
            "mvexpand": {
                "name": "mvexpand",
                "category": "transformation",
                "syntax": "mvexpand <field>",
                "description": "Expands multivalue field into separate events. Creates one event per value.",
                "examples": [
                    '| mvexpand tags',
                    '| makemv delim="," recipients | mvexpand recipients'
                ],
                "tips": [
                    "Creates duplicate events for each value",
                    "Other fields are duplicated",
                    "Useful before stats on multivalue fields"
                ],
                "related": ["makemv", "eval mvcount"]
            },
            "inputlookup": {
                "name": "inputlookup",
                "category": "data",
                "syntax": "inputlookup <lookup-name>",
                "description": "Loads all data from a lookup table as search results. Useful for viewing or processing lookup contents.",
                "examples": [
                    '| inputlookup user_info',
                    '| inputlookup assets.csv | search department="IT"',
                    '| inputlookup geo_ip | stats count by country'
                ],
                "tips": [
                    "Starts a new search from lookup data",
                    "Can be filtered with 'search' or 'where'",
                    "Use 'lookup' for enrichment instead"
                ],
                "related": ["lookup", "outputlookup"]
            },
            "outputlookup": {
                "name": "outputlookup",
                "category": "data",
                "syntax": "outputlookup <lookup-name>",
                "description": "Writes search results to a lookup table. Creates or updates the lookup file.",
                "examples": [
                    '| outputlookup threat_intel.csv',
                    '| stats count by user | outputlookup user_activity.csv',
                    '| outputlookup append=true new_entries.csv'
                ],
                "tips": [
                    "Creates new lookup or overwrites existing",
                    "Use append=true to add to existing lookup",
                    "Requires write permissions"
                ],
                "related": ["inputlookup", "lookup"]
            },
            "addinfo": {
                "name": "addinfo",
                "category": "metadata",
                "syntax": "addinfo",
                "description": "Adds search metadata fields to events including time range of the search.",
                "examples": [
                    '| addinfo',
                    '| addinfo | eval search_range=info_max_time-info_min_time'
                ],
                "tips": [
                    "Adds info_min_time, info_max_time, info_sid, etc.",
                    "Useful for calculating search time range",
                    "info_search_time is when the search ran"
                ],
                "related": ["metadata"]
            },
            "collect": {
                "name": "collect",
                "category": "data",
                "syntax": "collect index=<index> [sourcetype=<sourcetype>]",
                "description": "Writes search results to a specified index. Useful for creating summary indexes.",
                "examples": [
                    '| stats count by host | collect index=summary sourcetype=daily_host_counts',
                    '| collect index=notable marker="alert_rule_1"'
                ],
                "tips": [
                    "Creates new events in the specified index",
                    "Use for summary indexing / report acceleration",
                    "Requires write permissions to target index"
                ],
                "related": ["outputlookup", "sichart", "sitimechart"]
            },
            "multisearch": {
                "name": "multisearch",
                "category": "search",
                "syntax": "| multisearch [search ...] [search ...]",
                "description": "Runs multiple searches simultaneously and combines results. More efficient than append for multiple searches.",
                "examples": [
                    '| multisearch [search index=web status=500] [search index=app error=true]',
                    '| multisearch [search index=a] [search index=b] [search index=c]'
                ],
                "tips": [
                    "More efficient than multiple append commands",
                    "Results are combined, not matched",
                    "Each subsearch is independent"
                ],
                "related": ["append", "union"]
            },
            "tstats": {
                "name": "tstats",
                "category": "aggregation",
                "syntax": "| tstats <stats-function> from <datamodel> where <conditions> by <fields>",
                "description": "Fast statistical queries against indexed data or data models. Much faster than stats for supported operations.",
                "examples": [
                    '| tstats count where index=main by host',
                    '| tstats count from datamodel=Authentication where Authentication.action=failure by Authentication.user',
                    '| tstats summariesonly=true count from datamodel=Web by Web.status'
                ],
                "tips": [
                    "Much faster than regular stats",
                    "Works on tsidx (indexed) data",
                    "Best with data models and accelerated data",
                    "Limited functions compared to stats"
                ],
                "related": ["stats", "datamodel"]
            },
        }

    def _build_function_knowledge(self) -> Dict[str, Dict]:
        """Build knowledge base for SPL eval functions."""
        return {
            # Conditional Functions
            "if": {
                "name": "if",
                "category": "conditional",
                "syntax": "if(condition, true_value, false_value)",
                "description": "Returns true_value if condition is true, otherwise false_value. Basic conditional function.",
                "examples": [
                    'eval status_text = if(status<400, "success", "error")',
                    'eval is_admin = if(role="admin" OR role="superuser", 1, 0)',
                    'eval risk = if(severity>7, "high", if(severity>4, "medium", "low"))'
                ],
                "tips": [
                    "Can be nested for multiple conditions",
                    "Use 'case' for many conditions instead of nested if",
                    "Both true and false values are required"
                ],
                "related": ["case", "coalesce", "nullif"]
            },
            "case": {
                "name": "case",
                "category": "conditional",
                "syntax": "case(condition1, value1, condition2, value2, ..., true(), default)",
                "description": "Evaluates conditions in order and returns the value for the first true condition. Like a switch statement.",
                "examples": [
                    'eval priority = case(severity>=8, "critical", severity>=5, "high", severity>=3, "medium", true(), "low")',
                    'eval category = case(match(url, "^/api"), "api", match(url, "^/admin"), "admin", true(), "other")',
                    'eval day_type = case(strftime(_time, "%u")<6, "weekday", true(), "weekend")'
                ],
                "tips": [
                    "Conditions are evaluated in order",
                    "Use true() as the last condition for default",
                    "More readable than nested if statements"
                ],
                "related": ["if", "match", "like"]
            },
            "coalesce": {
                "name": "coalesce",
                "category": "conditional",
                "syntax": "coalesce(field1, field2, ..., default)",
                "description": "Returns the first non-null value from the arguments. Great for handling missing data.",
                "examples": [
                    'eval user = coalesce(username, user_id, "unknown")',
                    'eval ip = coalesce(src_ip, client_ip, remote_addr)',
                    'eval message = coalesce(error_msg, warning_msg, info_msg, "no message")'
                ],
                "tips": [
                    "Useful for normalizing fields with different names",
                    "Last argument can be a default value",
                    "Checks for null, not empty string"
                ],
                "related": ["if", "null", "nullif"]
            },
            "null": {
                "name": "null",
                "category": "conditional",
                "syntax": "null()",
                "description": "Returns null. Use to explicitly set a field to null.",
                "examples": [
                    'eval temp_field = null()',
                    'eval sensitive_data = if(redact=1, null(), sensitive_data)'
                ],
                "tips": [
                    "Takes no arguments",
                    "Use with 'where isnotnull()' to filter"
                ],
                "related": ["isnull", "isnotnull", "coalesce"]
            },
            # String Functions
            "len": {
                "name": "len",
                "category": "string",
                "syntax": "len(string)",
                "description": "Returns the character length of a string value.",
                "examples": [
                    'eval msg_length = len(message)',
                    '| where len(user) > 20',
                    'eval is_long = if(len(description) > 1000, "yes", "no")'
                ],
                "tips": [
                    "Returns null if field is null",
                    "Counts characters, not bytes"
                ],
                "related": ["substr", "lower", "upper"]
            },
            "lower": {
                "name": "lower",
                "category": "string",
                "syntax": "lower(string)",
                "description": "Converts string to lowercase. Useful for case-insensitive comparisons.",
                "examples": [
                    'eval user_lower = lower(user)',
                    '| where lower(action) = "login"',
                    'eval normalized_status = lower(status_text)'
                ],
                "tips": [
                    "Use for case-insensitive matching",
                    "SPL comparisons are case-sensitive by default"
                ],
                "related": ["upper", "trim"]
            },
            "upper": {
                "name": "upper",
                "category": "string",
                "syntax": "upper(string)",
                "description": "Converts string to uppercase.",
                "examples": [
                    'eval code = upper(error_code)',
                    'eval SEVERITY = upper(severity)'
                ],
                "tips": [
                    "Use for standardizing text values"
                ],
                "related": ["lower", "trim"]
            },
            "substr": {
                "name": "substr",
                "category": "string",
                "syntax": "substr(string, start, length)",
                "description": "Extracts a substring. Start position is 1-based.",
                "examples": [
                    'eval first_char = substr(code, 1, 1)',
                    'eval domain = substr(email, 1, len(email)-4)',
                    'eval year = substr(_time, 1, 4)'
                ],
                "tips": [
                    "Position 1 is the first character",
                    "Negative start counts from end",
                    "Length is optional (goes to end)"
                ],
                "related": ["len", "replace", "split"]
            },
            "replace": {
                "name": "replace",
                "category": "string",
                "syntax": "replace(string, regex, replacement)",
                "description": "Replaces text matching a regular expression with replacement string.",
                "examples": [
                    'eval clean_msg = replace(message, "password=\\w+", "password=***")',
                    'eval normalized = replace(path, "//+", "/")',
                    'eval no_digits = replace(text, "\\d", "")'
                ],
                "tips": [
                    "Uses regex, not literal strings",
                    "For literal replacement, escape special chars",
                    "Use \\1, \\2 for capture group references"
                ],
                "related": ["substr", "rex", "trim"]
            },
            "split": {
                "name": "split",
                "category": "string",
                "syntax": "split(string, delimiter)",
                "description": "Splits a string into a multivalue field using a delimiter.",
                "examples": [
                    'eval tags = split(tag_string, ",")',
                    'eval path_parts = split(url, "/")',
                    'eval words = split(message, " ")'
                ],
                "tips": [
                    "Creates multivalue field",
                    "Use mvindex to access specific elements",
                    "Use mvcount to count elements"
                ],
                "related": ["mvindex", "mvjoin", "mvcount"]
            },
            "trim": {
                "name": "trim",
                "category": "string",
                "syntax": "trim(string, trim_chars)",
                "description": "Removes leading and trailing characters. Default removes whitespace.",
                "examples": [
                    'eval clean = trim(value)',
                    "eval no_quotes = trim(field, '\"\\'')",
                    'eval trimmed = trim(data, " \\t\\n")'
                ],
                "tips": [
                    "Default trims whitespace",
                    "Use ltrim/rtrim for one-sided trim",
                    "Second arg is characters to remove, not a string"
                ],
                "related": ["ltrim", "rtrim", "replace"]
            },
            "urldecode": {
                "name": "urldecode",
                "category": "string",
                "syntax": "urldecode(url)",
                "description": "Decodes URL-encoded strings (converts %XX to characters).",
                "examples": [
                    'eval decoded_url = urldecode(url)',
                    'eval query = urldecode(query_string)'
                ],
                "tips": [
                    "Converts %20 to space, %3D to =, etc.",
                    "Useful for analyzing web logs"
                ],
                "related": ["replace"]
            },
            # Numeric Functions
            "tonumber": {
                "name": "tonumber",
                "category": "conversion",
                "syntax": "tonumber(string, base)",
                "description": "Converts a string to a number. Optional base for non-decimal.",
                "examples": [
                    'eval bytes_num = tonumber(bytes)',
                    'eval hex_value = tonumber(hex_string, 16)',
                    '| where tonumber(status) >= 400'
                ],
                "tips": [
                    "Returns null if conversion fails",
                    "Base 16 for hex, 8 for octal, 2 for binary",
                    "Default base is 10"
                ],
                "related": ["tostring", "round", "floor", "ceil"]
            },
            "tostring": {
                "name": "tostring",
                "category": "conversion",
                "syntax": "tostring(value, format)",
                "description": "Converts a value to string. Format options for numbers and time.",
                "examples": [
                    'eval time_str = tostring(_time, "hex")',
                    'eval num_str = tostring(count)',
                    'eval formatted = tostring(bytes, "commas")',
                    'eval duration = tostring(seconds, "duration")'
                ],
                "tips": [
                    '"hex" for hexadecimal',
                    '"commas" for thousand separators',
                    '"duration" for human-readable time'
                ],
                "related": ["tonumber", "strftime"]
            },
            "round": {
                "name": "round",
                "category": "math",
                "syntax": "round(number, precision)",
                "description": "Rounds a number to specified decimal places.",
                "examples": [
                    'eval rounded = round(avg_time, 2)',
                    'eval whole = round(value)',
                    'eval percent = round(ratio * 100, 1)'
                ],
                "tips": [
                    "Default precision is 0 (whole number)",
                    "Negative precision rounds to tens, hundreds, etc."
                ],
                "related": ["floor", "ceil", "abs"]
            },
            "floor": {
                "name": "floor",
                "category": "math",
                "syntax": "floor(number)",
                "description": "Rounds down to the nearest integer.",
                "examples": [
                    'eval whole = floor(decimal_value)',
                    'eval hours = floor(seconds / 3600)'
                ],
                "tips": [
                    "Always rounds toward negative infinity",
                    "floor(3.9) = 3, floor(-3.1) = -4"
                ],
                "related": ["ceil", "round"]
            },
            "ceil": {
                "name": "ceil",
                "category": "math",
                "syntax": "ceil(number) or ceiling(number)",
                "description": "Rounds up to the nearest integer.",
                "examples": [
                    'eval pages = ceil(total_items / items_per_page)',
                    'eval rounded_up = ceil(value)'
                ],
                "tips": [
                    "Always rounds toward positive infinity",
                    "ceil(3.1) = 4, ceil(-3.9) = -3"
                ],
                "related": ["floor", "round"]
            },
            "abs": {
                "name": "abs",
                "category": "math",
                "syntax": "abs(number)",
                "description": "Returns the absolute (positive) value of a number.",
                "examples": [
                    'eval difference = abs(expected - actual)',
                    'eval magnitude = abs(value)'
                ],
                "tips": [
                    "Useful for calculating differences regardless of direction"
                ],
                "related": ["round", "floor", "ceil"]
            },
            "pow": {
                "name": "pow",
                "category": "math",
                "syntax": "pow(base, exponent)",
                "description": "Returns base raised to the power of exponent.",
                "examples": [
                    'eval squared = pow(value, 2)',
                    'eval kb = bytes / pow(2, 10)',
                    'eval mb = bytes / pow(1024, 2)'
                ],
                "tips": [
                    "Use for exponential calculations",
                    "pow(2, 10) = 1024"
                ],
                "related": ["sqrt", "log", "exp"]
            },
            "sqrt": {
                "name": "sqrt",
                "category": "math",
                "syntax": "sqrt(number)",
                "description": "Returns the square root of a number.",
                "examples": [
                    'eval root = sqrt(value)',
                    'eval distance = sqrt(pow(x2-x1, 2) + pow(y2-y1, 2))'
                ],
                "tips": [
                    "Returns null for negative numbers"
                ],
                "related": ["pow", "log"]
            },
            # Time Functions
            "now": {
                "name": "now",
                "category": "time",
                "syntax": "now()",
                "description": "Returns the current epoch time (seconds since 1970-01-01).",
                "examples": [
                    'eval current_time = now()',
                    'eval age_seconds = now() - _time',
                    'eval is_recent = if(now() - _time < 3600, "yes", "no")'
                ],
                "tips": [
                    "Returns Unix timestamp",
                    "Use with _time for age calculations",
                    "Use strftime to format output"
                ],
                "related": ["time", "relative_time", "strftime"]
            },
            "time": {
                "name": "time",
                "category": "time",
                "syntax": "time()",
                "description": "Returns the current time as epoch. Same as now().",
                "examples": [
                    'eval current = time()'
                ],
                "tips": [
                    "Alias for now()"
                ],
                "related": ["now", "relative_time"]
            },
            "relative_time": {
                "name": "relative_time",
                "category": "time",
                "syntax": "relative_time(timestamp, modifier)",
                "description": "Adjusts a timestamp by a relative amount. Modifier syntax like -1h, +7d, @d.",
                "examples": [
                    'eval yesterday = relative_time(now(), "-1d@d")',
                    'eval week_ago = relative_time(_time, "-7d")',
                    'eval start_of_day = relative_time(_time, "@d")',
                    'eval start_of_month = relative_time(now(), "@mon")'
                ],
                "tips": [
                    "@ snaps to boundary (e.g., @d = start of day)",
                    "Common modifiers: s, m, h, d, w, mon, y",
                    "Combine: -1d@d means yesterday at midnight"
                ],
                "related": ["now", "strftime", "strptime"]
            },
            "strftime": {
                "name": "strftime",
                "category": "time",
                "syntax": 'strftime(timestamp, "format")',
                "description": "Formats a timestamp as a human-readable string. Essential for time display.",
                "examples": [
                    'eval date = strftime(_time, "%Y-%m-%d")',
                    'eval datetime = strftime(_time, "%Y-%m-%d %H:%M:%S")',
                    'eval day_name = strftime(_time, "%A")',
                    'eval hour = strftime(_time, "%H")',
                    'eval month = strftime(_time, "%B %Y")'
                ],
                "tips": [
                    "%Y=year, %m=month, %d=day",
                    "%H=hour(24), %I=hour(12), %M=minute, %S=second",
                    "%A=day name, %B=month name",
                    "%u=day of week (1-7, Mon=1)"
                ],
                "related": ["strptime", "now", "relative_time"]
            },
            "strptime": {
                "name": "strptime",
                "category": "time",
                "syntax": 'strptime(string, "format")',
                "description": "Parses a time string into epoch timestamp. Opposite of strftime.",
                "examples": [
                    'eval epoch = strptime(date_string, "%Y-%m-%d")',
                    'eval timestamp = strptime(datetime, "%Y-%m-%d %H:%M:%S")',
                    'eval log_time = strptime(log_date, "%b %d %Y %H:%M:%S")'
                ],
                "tips": [
                    "Format must match string exactly",
                    "Returns epoch timestamp",
                    "Use same format codes as strftime"
                ],
                "related": ["strftime", "now"]
            },
            # Multivalue Functions
            "mvcount": {
                "name": "mvcount",
                "category": "multivalue",
                "syntax": "mvcount(field)",
                "description": "Returns the number of values in a multivalue field.",
                "examples": [
                    'eval tag_count = mvcount(tags)',
                    '| where mvcount(recipients) > 5',
                    'eval has_multiple = if(mvcount(values) > 1, "yes", "no")'
                ],
                "tips": [
                    "Returns null for null/empty fields",
                    "Returns 1 for single-value fields"
                ],
                "related": ["mvindex", "mvjoin", "mvfind"]
            },
            "mvindex": {
                "name": "mvindex",
                "category": "multivalue",
                "syntax": "mvindex(field, start, end)",
                "description": "Returns value(s) at specified index in a multivalue field. 0-based indexing.",
                "examples": [
                    'eval first = mvindex(values, 0)',
                    'eval last = mvindex(values, -1)',
                    'eval first_three = mvindex(values, 0, 2)',
                    'eval domain = mvindex(split(email, "@"), 1)'
                ],
                "tips": [
                    "Index 0 is the first value",
                    "Negative index counts from end (-1 is last)",
                    "Optional end returns range of values"
                ],
                "related": ["mvcount", "mvfind", "split"]
            },
            "mvjoin": {
                "name": "mvjoin",
                "category": "multivalue",
                "syntax": "mvjoin(field, delimiter)",
                "description": "Joins multivalue field into a single string with delimiter.",
                "examples": [
                    'eval tag_list = mvjoin(tags, ", ")',
                    'eval path = mvjoin(path_parts, "/")',
                    'eval csv_line = mvjoin(values, ",")'
                ],
                "tips": [
                    "Opposite of split()",
                    "Useful for display formatting"
                ],
                "related": ["split", "mvcount", "mvindex"]
            },
            "mvfind": {
                "name": "mvfind",
                "category": "multivalue",
                "syntax": "mvfind(field, regex)",
                "description": "Returns index of first value matching regex, or null if not found.",
                "examples": [
                    'eval admin_pos = mvfind(users, "^admin")',
                    'eval has_error = if(isnotnull(mvfind(tags, "error")), "yes", "no")'
                ],
                "tips": [
                    "Returns index (0-based), not the value",
                    "Use to check if pattern exists in multivalue"
                ],
                "related": ["mvindex", "mvcount", "match"]
            },
            "mvfilter": {
                "name": "mvfilter",
                "category": "multivalue",
                "syntax": "mvfilter(expression)",
                "description": "Filters multivalue field to keep only values matching expression.",
                "examples": [
                    'eval errors = mvfilter(match(messages, "error"))',
                    'eval high_values = mvfilter(values > 100)',
                    'eval valid_ips = mvfilter(like(ips, "10.%"))'
                ],
                "tips": [
                    "Expression is evaluated for each value",
                    "Use with match, like, or comparisons",
                    "Returns multivalue with matching values"
                ],
                "related": ["mvfind", "mvindex", "match"]
            },
            # Informational Functions
            "isnull": {
                "name": "isnull",
                "category": "informational",
                "syntax": "isnull(field)",
                "description": "Returns true if field is null.",
                "examples": [
                    '| where isnull(user)',
                    'eval has_user = if(isnull(user), "no", "yes")',
                    'eval status = if(isnull(error), "success", "failed")'
                ],
                "tips": [
                    "Checks for null, not empty string",
                    "Use for missing field detection"
                ],
                "related": ["isnotnull", "null", "coalesce"]
            },
            "isnotnull": {
                "name": "isnotnull",
                "category": "informational",
                "syntax": "isnotnull(field)",
                "description": "Returns true if field is not null.",
                "examples": [
                    '| where isnotnull(src_ip)',
                    'eval has_data = if(isnotnull(response), 1, 0)'
                ],
                "tips": [
                    "Opposite of isnull",
                    "Commonly used in where filters"
                ],
                "related": ["isnull", "null", "coalesce"]
            },
            "isnum": {
                "name": "isnum",
                "category": "informational",
                "syntax": "isnum(value)",
                "description": "Returns true if value is a number.",
                "examples": [
                    '| where isnum(port)',
                    'eval is_numeric = if(isnum(value), "number", "string")'
                ],
                "tips": [
                    "Useful for validation before math operations"
                ],
                "related": ["isint", "isstr", "typeof"]
            },
            "isstr": {
                "name": "isstr",
                "category": "informational",
                "syntax": "isstr(value)",
                "description": "Returns true if value is a string.",
                "examples": [
                    '| where isstr(message)',
                    'eval type = if(isstr(value), "text", "other")'
                ],
                "tips": [
                    "Useful for type checking"
                ],
                "related": ["isnum", "isint", "typeof"]
            },
            "typeof": {
                "name": "typeof",
                "category": "informational",
                "syntax": "typeof(value)",
                "description": "Returns the data type of a value as a string.",
                "examples": [
                    'eval field_type = typeof(value)',
                    '| stats count by typeof(data)'
                ],
                "tips": [
                    "Returns: Number, String, Boolean, etc.",
                    "Useful for debugging data types"
                ],
                "related": ["isnum", "isstr", "isnull"]
            },
            # Pattern Matching
            "match": {
                "name": "match",
                "category": "comparison",
                "syntax": "match(string, regex)",
                "description": "Returns true if string matches the regular expression.",
                "examples": [
                    '| where match(email, "^[a-z]+@company\\.com$")',
                    'eval is_ip = if(match(value, "^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$"), 1, 0)',
                    'eval has_error = match(message, "(?i)error|fail|exception")'
                ],
                "tips": [
                    "Uses regex syntax",
                    "(?i) makes case-insensitive",
                    "More flexible than like()"
                ],
                "related": ["like", "regex", "rex"]
            },
            "like": {
                "name": "like",
                "category": "comparison",
                "syntax": "like(string, pattern)",
                "description": "Returns true if string matches pattern. Uses % for wildcards.",
                "examples": [
                    '| where like(user, "admin%")',
                    'eval is_internal = if(like(ip, "10.%") OR like(ip, "192.168.%"), "yes", "no")',
                    '| where like(path, "%/api/%")'
                ],
                "tips": [
                    "% matches any characters",
                    "_ matches single character",
                    "Simpler than regex for basic patterns"
                ],
                "related": ["match", "cidrmatch"]
            },
            "cidrmatch": {
                "name": "cidrmatch",
                "category": "comparison",
                "syntax": 'cidrmatch("cidr", ip)',
                "description": "Returns true if IP address is within the CIDR range.",
                "examples": [
                    '| where cidrmatch("10.0.0.0/8", src_ip)',
                    'eval is_internal = if(cidrmatch("192.168.0.0/16", ip) OR cidrmatch("10.0.0.0/8", ip), "internal", "external")',
                    'eval network = case(cidrmatch("10.1.0.0/16", ip), "datacenter", cidrmatch("10.2.0.0/16", ip), "office", true(), "other")'
                ],
                "tips": [
                    "CIDR must be a string in quotes",
                    "Supports IPv4 CIDR notation",
                    "Efficient for IP range matching"
                ],
                "related": ["like", "match"]
            },
            # Cryptographic Functions
            "md5": {
                "name": "md5",
                "category": "cryptographic",
                "syntax": "md5(string)",
                "description": "Returns MD5 hash of the string.",
                "examples": [
                    'eval hash = md5(message)',
                    'eval file_id = md5(path . size)'
                ],
                "tips": [
                    "Returns 32-character hex string",
                    "Use for checksums, not security"
                ],
                "related": ["sha1", "sha256", "sha512"]
            },
            "sha1": {
                "name": "sha1",
                "category": "cryptographic",
                "syntax": "sha1(string)",
                "description": "Returns SHA-1 hash of the string.",
                "examples": [
                    'eval hash = sha1(content)',
                    'eval unique_id = sha1(user . session)'
                ],
                "tips": [
                    "Returns 40-character hex string",
                    "Stronger than MD5"
                ],
                "related": ["md5", "sha256", "sha512"]
            },
            "sha256": {
                "name": "sha256",
                "category": "cryptographic",
                "syntax": "sha256(string)",
                "description": "Returns SHA-256 hash of the string.",
                "examples": [
                    'eval hash = sha256(file_content)',
                    'eval secure_id = sha256(sensitive_data)'
                ],
                "tips": [
                    "Returns 64-character hex string",
                    "Recommended for most hashing needs"
                ],
                "related": ["md5", "sha1", "sha512"]
            },
        }

    def _build_concept_knowledge(self) -> List[Dict]:
        """Build knowledge base for SPL concepts."""
        return [
            {
                "name": "Search Pipeline",
                "category": "core_concept",
                "description": "SPL queries are pipelines of commands connected by the pipe (|) character. Data flows left to right, with each command transforming the results before passing them to the next command.",
                "content": """
The SPL pipeline is the fundamental structure of every Splunk query:

search terms | command1 | command2 | command3

Key points:
- Data flows from left to right
- Each command receives results from the previous command
- Each command can filter, transform, or aggregate the data
- The first command (implicit 'search') retrieves initial events
- Subsequent commands process those events

Example pipeline:
index=web status>=400 
| stats count by status, uri 
| sort -count 
| head 10

This pipeline:
1. Searches for web events with status >= 400
2. Counts occurrences by status and URI
3. Sorts by count descending
4. Returns top 10 results
""",
                "keywords": ["pipeline", "pipe", "command", "flow", "transform"]
            },
            {
                "name": "Time Ranges",
                "category": "core_concept",
                "description": "Splunk searches always operate within a time range. You can specify times using relative modifiers, absolute dates, or the time picker.",
                "content": """
Time is critical in Splunk searches. Every search has a time range.

Relative Time Modifiers:
- -1h = 1 hour ago
- -24h or -1d = 24 hours ago
- -7d or -1w = 7 days ago
- -30d = 30 days ago
- @d = snap to start of day
- @h = snap to start of hour

Examples:
earliest=-24h latest=now    # Last 24 hours
earliest=-7d@d latest=@d    # Last 7 days, aligned to midnight
earliest=-1h@h              # From start of last hour

In queries:
index=main earliest=-1h
index=security earliest=-7d latest=-1d

Time Field (_time):
- Every event has _time (epoch timestamp)
- Use strftime(_time, "%Y-%m-%d") to format
- Use relative_time() for calculations
""",
                "keywords": ["time", "earliest", "latest", "relative_time", "_time", "timerange"]
            },
            {
                "name": "Indexes and Sourcetypes",
                "category": "core_concept",
                "description": "Indexes are containers for your data. Sourcetypes identify the format of data. Always specify both when possible for efficient searches.",
                "content": """
Indexes and sourcetypes are fundamental to organizing Splunk data.

Indexes:
- Logical containers for data
- Like database tables
- Specify with index=<name>
- Use index=* to search all (slow!)

Common indexes:
- main (default)
- security, web, network
- wazuh-alerts (for Wazuh)

Sourcetypes:
- Identify data format
- Determine how data is parsed
- Specify with sourcetype=<name>

Best practices:
- ALWAYS specify index when possible
- Add sourcetype if you know it
- Avoid index=* in production

Example:
index=security sourcetype=linux_secure
index=web sourcetype=access_combined
index=wazuh-alerts sourcetype=wazuh-alerts
""",
                "keywords": ["index", "sourcetype", "data", "search", "performance"]
            },
            {
                "name": "Fields and Field Extraction",
                "category": "core_concept",
                "description": "Fields are name-value pairs extracted from events. Splunk automatically extracts some fields, and you can extract more using rex or field extractions.",
                "content": """
Fields are the building blocks of SPL queries.

Default Fields (always available):
- _time: Event timestamp
- _raw: Original raw event text
- host: Source host
- source: Data source path
- sourcetype: Data format type
- index: Index containing the event

Field Extraction Methods:
1. Automatic: Splunk extracts key=value pairs
2. rex command: Extract with regex
3. Field extractions: Configured in Splunk
4. eval: Calculate/derive new fields

Using rex for extraction:
| rex field=_raw "user=(?<username>\\w+)"
| rex field=message "error code: (?<error_code>\\d+)"

Field naming conventions:
- Case-sensitive
- Can contain letters, numbers, underscores
- Avoid spaces and special characters
""",
                "keywords": ["fields", "extraction", "rex", "_raw", "_time", "parse"]
            },
            {
                "name": "Stats vs Eventstats vs Streamstats",
                "category": "core_concept",
                "description": "Understanding the differences between stats, eventstats, and streamstats is crucial for effective aggregation.",
                "content": """
Three related but different aggregation commands:

STATS:
- Collapses events into summary rows
- Events are consumed, only stats remain
- Use for: Totals, summaries, groupings

| stats count by user
Result: One row per user with their count

EVENTSTATS:
- Adds stats to each event WITHOUT collapsing
- Original events preserved with stats added
- Use for: Comparing events to group averages

| eventstats avg(bytes) as avg_bytes by host
Result: Every event gets avg_bytes field for its host

STREAMSTATS:
- Running/cumulative calculations as events stream
- Processes events in order they appear
- Use for: Running totals, moving averages

| sort _time | streamstats count as running_total
Result: Each event numbered in sequence

| streamstats avg(value) as moving_avg window=5
Result: 5-event moving average
""",
                "keywords": ["stats", "eventstats", "streamstats", "aggregation", "running"]
            },
            {
                "name": "Subsearches",
                "category": "core_concept",
                "description": "Subsearches are searches within searches, enclosed in square brackets. They run first and their results are used by the outer search.",
                "content": """
Subsearches allow you to use results from one search in another.

Syntax: [search ...]

How they work:
1. Subsearch runs first
2. Results become part of outer search
3. Typically returns field values for filtering

Basic example:
index=main [search index=alerts | fields src_ip]

This:
1. Searches alerts index
2. Gets all src_ip values
3. Uses them to filter main index

Common patterns:

Filtering by lookup results:
index=web [search index=threats | fields ip]

Getting a dynamic value:
index=main [search index=config | return $value]

With join:
index=main | join user [search index=users | fields user, department]

Subsearch limitations:
- Default: 60 seconds timeout
- Default: 50,000 results max
- Can slow down searches if large

Best practices:
- Keep subsearches small and fast
- Use 'fields' to limit returned fields
- Consider 'lookup' for static data
""",
                "keywords": ["subsearch", "join", "filter", "nested", "brackets"]
            },
            {
                "name": "Multivalue Fields",
                "category": "core_concept",
                "description": "Multivalue fields contain multiple values in a single field. They require special functions to work with properly.",
                "content": """
Multivalue fields store multiple values in one field.

Creating multivalue fields:
- makemv: Split string into multivalue
- split(): Eval function to split
- values()/list(): Stats aggregation

Working with multivalue:
| eval emails = split(recipients, ",")
| eval count = mvcount(emails)
| eval first = mvindex(emails, 0)
| eval last = mvindex(emails, -1)
| eval joined = mvjoin(emails, "; ")

Key functions:
- mvcount(): Count values
- mvindex(field, n): Get nth value (0-based)
- mvjoin(field, delim): Join into string
- mvfind(field, regex): Find matching value index
- mvfilter(expr): Filter values
- mvexpand: Create separate events per value

Example: Finding events with specific tag:
| makemv delim="," tags
| where mvfind(tags, "critical") >= 0

Example: Expanding to analyze each value:
| makemv delim="," items
| mvexpand items
| stats count by items
""",
                "keywords": ["multivalue", "mvcount", "mvindex", "mvjoin", "mvexpand", "split"]
            },
            {
                "name": "Transactions",
                "category": "core_concept",
                "description": "Transactions group related events together based on common fields and time proximity. Useful for session analysis and correlating events.",
                "content": """
Transactions group events that belong together logically.

Basic syntax:
| transaction <fields> [options]

Common options:
- maxspan=<time>: Max duration of transaction
- maxpause=<time>: Max gap between events
- startswith=<search>: Event that starts transaction
- endswith=<search>: Event that ends transaction

Example: Session analysis
| transaction session_id maxspan=30m

Example: Login to logout
| transaction user startswith="login" endswith="logout"

Fields added by transaction:
- duration: Time span of transaction
- eventcount: Number of events
- _time: Earliest event time

Use cases:
- Web sessions
- Login/logout tracking
- Request/response correlation
- Multi-step processes

Performance note:
Transactions are memory-intensive. For large datasets, consider:
| stats earliest(_time) as start, latest(_time) as end, count by session_id
| eval duration = end - start

This is often faster than transaction.
""",
                "keywords": ["transaction", "session", "correlation", "group", "duration"]
            },
            {
                "name": "Lookups",
                "category": "core_concept",
                "description": "Lookups enrich your events with data from external tables (CSV files, KV stores, etc.). Much more efficient than joins for static data.",
                "content": """
Lookups add context to your events from reference data.

Types of lookups:
1. CSV lookups: Static CSV files
2. KV Store: Key-value database
3. External: Scripts/commands

Using lookups:
| lookup <lookup_name> <match_field> [OUTPUT <fields>]

Example:
| lookup user_info username OUTPUT department, full_name
| lookup geo_ip ip OUTPUT country, city
| lookup asset_inventory ip as src_ip OUTPUTNEW hostname

INPUT vs OUTPUT:
- Match fields: Fields to look up by
- OUTPUT: Fields to add (overwrites existing)
- OUTPUTNEW: Fields to add (keeps existing)

Viewing lookup contents:
| inputlookup user_info.csv
| inputlookup user_info.csv | search department="IT"

Creating/updating lookups:
| stats count by user | outputlookup user_counts.csv

Lookup vs Join:
- Lookup: For static reference data, very fast
- Join: For dynamic data from another search

Best practices:
- Use lookups for enrichment
- Keep lookup tables updated
- Use OUTPUTNEW to preserve existing values
""",
                "keywords": ["lookup", "enrich", "csv", "inputlookup", "outputlookup", "reference"]
            },
            {
                "name": "Search Optimization",
                "category": "best_practices",
                "description": "Tips for writing efficient Splunk searches that run fast and use minimal resources.",
                "content": """
Key principles for fast, efficient searches:

1. Be Specific Early
   - Specify index and sourcetype
   - Add time range (earliest/latest)
   - Filter with search terms before pipe

   Good: index=web sourcetype=access status>=400 earliest=-1h
   Bad:  index=* | search status>=400

2. Use Indexed Fields
   - Fields like host, source, sourcetype are indexed
   - Searching indexed fields is very fast
   - Custom indexed fields depend on your config

3. Reduce Data Early
   - Filter before transforming
   - Remove unneeded fields with 'fields -'
   - Use 'where' instead of 'search' after stats

4. Avoid Expensive Operations
   - Minimize wildcards (especially leading *)
   - Avoid 'transaction' on large datasets
   - Use 'tstats' instead of 'stats' when possible
   - Limit subsearch results

5. Use Efficient Commands
   - 'stats' is faster than 'transaction'
   - 'tstats' is faster than 'stats'
   - 'lookup' is faster than 'join'

6. Summary Indexing
   - Pre-compute common aggregations
   - Use report acceleration
   - Consider data models for frequent queries
""",
                "keywords": ["performance", "optimization", "fast", "efficient", "slow", "timeout"]
            },
        ]

    def _build_example_knowledge(self) -> List[Dict]:
        """Build common SPL query examples."""
        return [
            {
                "name": "Top N by Count",
                "description": "Find the most common values of a field",
                "query": "| top 10 user",
                "equivalent": "| stats count by user | sort -count | head 10",
                "use_case": "Finding most active users, common errors, frequent IPs",
                "keywords": ["top", "most", "common", "frequent", "count"]
            },
            {
                "name": "Events Over Time",
                "description": "Count events in time buckets",
                "query": "| timechart span=1h count",
                "variants": [
                    "| timechart span=5m count by status",
                    "| timechart span=1d sum(bytes)",
                ],
                "use_case": "Trend analysis, monitoring patterns",
                "keywords": ["timeline", "trend", "over time", "chart", "graph"]
            },
            {
                "name": "Unique Value Count",
                "description": "Count distinct/unique values",
                "query": "| stats dc(user) as unique_users",
                "variants": [
                    "| stats dc(src_ip) as unique_ips by host",
                    "| timechart span=1h dc(session_id) as sessions",
                ],
                "use_case": "Unique visitors, distinct IPs, cardinality",
                "keywords": ["unique", "distinct", "different", "cardinality", "dc"]
            },
            {
                "name": "Percentage of Total",
                "description": "Calculate percentage of each group",
                "query": "| stats count by status | eventstats sum(count) as total | eval percent=round(count/total*100, 2)",
                "use_case": "Distribution analysis, proportions",
                "keywords": ["percentage", "percent", "ratio", "distribution", "proportion"]
            },
            {
                "name": "First and Last Events",
                "description": "Get earliest and latest events",
                "query": "| stats earliest(_time) as first_seen, latest(_time) as last_seen by user",
                "use_case": "Session boundaries, activity windows",
                "keywords": ["first", "last", "earliest", "latest", "start", "end"]
            },
            {
                "name": "Events Within Time Window",
                "description": "Find events in a specific time range",
                "query": "earliest=-24h latest=now | ...",
                "variants": [
                    "earliest=-7d@d latest=@d  # Last 7 complete days",
                    "earliest=-1h@h            # From start of last hour",
                ],
                "use_case": "Historical analysis, specific time windows",
                "keywords": ["time", "range", "between", "during", "period"]
            },
            {
                "name": "Filter by Multiple Values",
                "description": "Match any of several values",
                "query": '| search status IN (400, 401, 403, 404, 500)',
                "variants": [
                    '| where status IN (400, 401, 403)',
                    '| search (status=400 OR status=401 OR status=403)',
                ],
                "use_case": "Filtering by error codes, specific statuses",
                "keywords": ["multiple", "several", "list", "any", "IN", "OR"]
            },
            {
                "name": "Pattern Matching",
                "description": "Filter by pattern in field",
                "query": '| where like(user, "admin%")',
                "variants": [
                    '| regex user="^admin"',
                    '| where match(email, ".*@company\\.com$")',
                ],
                "use_case": "Finding users by pattern, domain filtering",
                "keywords": ["pattern", "like", "match", "regex", "wildcard", "contains"]
            },
            {
                "name": "Group Statistics",
                "description": "Multiple aggregations grouped by field",
                "query": "| stats count, avg(response_time), max(response_time), min(response_time) by endpoint",
                "use_case": "Performance analysis, summary statistics",
                "keywords": ["group", "summary", "aggregate", "statistics", "by"]
            },
            {
                "name": "Calculate Duration",
                "description": "Calculate time between events",
                "query": "| transaction session_id | eval duration_min=duration/60",
                "variants": [
                    "| stats earliest(_time) as start, latest(_time) as end by session | eval duration=end-start",
                ],
                "use_case": "Session duration, processing time",
                "keywords": ["duration", "time", "length", "how long", "elapsed"]
            },
        ]

    def _build_best_practices(self) -> List[Dict]:
        """Build SPL best practices."""
        return [
            {
                "title": "Always Specify Index",
                "category": "performance",
                "description": "Always include index= in your search. Searching all indexes (index=*) is slow and resource-intensive.",
                "good": 'index=security status=failed',
                "bad": 'status=failed',
                "impact": "Can improve search speed by 10x or more",
            },
            {
                "title": "Filter Early in Pipeline",
                "category": "performance",
                "description": "Put your most restrictive filters at the beginning of the search to reduce data processed by later commands.",
                "good": 'index=web status>=400 | stats count by uri',
                "bad": 'index=web | stats count by uri, status | where status>=400',
                "impact": "Reduces data processed by expensive operations",
            },
            {
                "title": "Use Fields Command",
                "category": "performance",
                "description": "Remove unnecessary fields early with 'fields -' to reduce memory usage and speed up processing.",
                "good": 'index=main | fields host, message | stats count by host',
                "bad": 'index=main | stats count by host',
                "impact": "Reduces memory usage, especially with _raw field",
            },
            {
                "title": "Prefer Stats Over Transaction",
                "category": "performance",
                "description": "For simple aggregations, stats is much faster than transaction. Only use transaction when you need the grouped events.",
                "good": '| stats earliest(_time) as start, latest(_time) as end, count by session_id',
                "bad": '| transaction session_id | stats count, avg(duration)',
                "impact": "Transaction can be 10-100x slower than stats",
            },
            {
                "title": "Use Lookup Instead of Join",
                "category": "performance",
                "description": "For enriching with static reference data, lookups are much more efficient than joins.",
                "good": '| lookup user_info username OUTPUT department',
                "bad": '| join username [search index=users | fields username, department]',
                "impact": "Lookups are orders of magnitude faster for static data",
            },
            {
                "title": "Avoid Leading Wildcards",
                "category": "performance",
                "description": "Wildcards at the start of search terms (*error) prevent index optimization and are slow.",
                "good": 'index=main error',
                "bad": 'index=main *error*',
                "impact": "Leading wildcards disable index optimization",
            },
            {
                "title": "Use Case Functions for Case-Insensitive",
                "category": "correctness",
                "description": "SPL comparisons are case-sensitive. Use lower() or upper() for case-insensitive matching.",
                "good": '| where lower(action)="login"',
                "bad": '| where action="login"  # Misses LOGIN, Login, etc.',
                "impact": "Ensures you find all matching events",
            },
            {
                "title": "Handle Null Values",
                "category": "correctness",
                "description": "Use coalesce(), isnull(), or fillnull to handle missing values in calculations.",
                "good": '| eval total = coalesce(count, 0)',
                "bad": '| eval total = count  # Null if count is missing',
                "impact": "Prevents null results and calculation errors",
            },
            {
                "title": "Use Appropriate Time Spans",
                "category": "visualization",
                "description": "Match timechart span to your time range. Too small = noisy, too large = loses detail.",
                "good": 'earliest=-24h | timechart span=1h count',
                "bad": 'earliest=-24h | timechart span=1m count  # 1440 buckets!',
                "impact": "Better visualizations and query performance",
            },
            {
                "title": "Name Your Fields",
                "category": "readability",
                "description": "Use 'as' to give meaningful names to calculated fields and aggregations.",
                "good": '| stats count as event_count, avg(bytes) as avg_size by host',
                "bad": '| stats count, avg(bytes) by host',
                "impact": "More readable results and dashboards",
            },
        ]

    def initialize(self) -> bool:
        """Initialize the RAG system."""
        if self.is_initialized:
            return True

        try:
            # Load custom data if exists
            if self.data_path.exists():
                with open(self.data_path) as f:
                    self.data = json.load(f)
                logger.info(f"Loaded SPL knowledge from {self.data_path}")
            else:
                logger.info("Using built-in SPL knowledge only")
                self.data = {}

            # Build document corpus
            self._build_documents()

            # Build embeddings
            self._build_embeddings()

            self.is_initialized = True
            logger.info(f"SPL RAG initialized with {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SPL RAG: {e}")
            return False

    def _build_documents(self):
        """Build the document corpus for semantic search."""
        self.documents = []

        # Add command documentation
        for cmd_name, cmd_data in self.commands.items():
            content = f"SPL Command: {cmd_name}\n"
            content += f"Category: {cmd_data.get('category', '')}\n"
            content += f"Syntax: {cmd_data.get('syntax', '')}\n"
            content += f"Description: {cmd_data.get('description', '')}\n"
            if cmd_data.get('examples'):
                content += f"Examples: {', '.join(cmd_data['examples'][:2])}\n"
            if cmd_data.get('tips'):
                content += f"Tips: {' '.join(cmd_data['tips'][:2])}"

            self.documents.append({
                "type": "command",
                "name": cmd_name,
                "content": content,
                "data": cmd_data
            })

        # Add function documentation
        for func_name, func_data in self.functions.items():
            content = f"SPL Function: {func_name}\n"
            content += f"Category: {func_data.get('category', '')}\n"
            content += f"Syntax: {func_data.get('syntax', '')}\n"
            content += f"Description: {func_data.get('description', '')}\n"
            if func_data.get('examples'):
                content += f"Examples: {', '.join(func_data['examples'][:2])}"

            self.documents.append({
                "type": "function",
                "name": func_name,
                "content": content,
                "data": func_data
            })

        # Add concepts
        for concept in self.concepts:
            content = f"SPL Concept: {concept['name']}\n"
            content += f"Category: {concept.get('category', '')}\n"
            content += f"Description: {concept.get('description', '')}\n"
            content += f"Keywords: {', '.join(concept.get('keywords', []))}"

            self.documents.append({
                "type": "concept",
                "name": concept["name"],
                "content": content,
                "data": concept
            })

        # Add examples
        for example in self.examples:
            content = f"SPL Example: {example['name']}\n"
            content += f"Description: {example.get('description', '')}\n"
            content += f"Query: {example.get('query', '')}\n"
            content += f"Use case: {example.get('use_case', '')}\n"
            content += f"Keywords: {', '.join(example.get('keywords', []))}"

            self.documents.append({
                "type": "example",
                "name": example["name"],
                "content": content,
                "data": example
            })

        # Add best practices
        for practice in self.best_practices:
            content = f"SPL Best Practice: {practice['title']}\n"
            content += f"Category: {practice.get('category', '')}\n"
            content += f"Description: {practice.get('description', '')}\n"
            if practice.get('good'):
                content += f"Good: {practice['good']}\n"
            if practice.get('bad'):
                content += f"Bad: {practice['bad']}"

            self.documents.append({
                "type": "best_practice",
                "name": practice["title"],
                "content": content,
                "data": practice
            })

    def _build_embeddings(self):
        """Build embeddings for semantic search."""
        model = get_embedding_model()
        if model is None:
            logger.warning("No embedding model available for SPL RAG")
            self.embeddings = None
            return

        try:
            texts = [doc["content"] for doc in self.documents]
            self.embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            logger.info(f"Built embeddings for {len(self.documents)} SPL documents")
        except Exception as e:
            logger.error(f"Failed to build SPL embeddings: {e}")
            self.embeddings = None

    def search(self, query: str, top_k: int = 10, doc_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for relevant SPL knowledge.

        Args:
            query: User's question about SPL
            top_k: Number of results to return
            doc_types: Filter by document types (command, function, concept, example, best_practice)

        Returns:
            List of relevant documents with similarity scores
        """
        if not self.is_initialized:
            self.initialize()

        if not self.documents:
            return []

        # Filter by document types if specified
        if doc_types:
            filtered_indices = [i for i, doc in enumerate(self.documents) if doc["type"] in doc_types]
        else:
            filtered_indices = list(range(len(self.documents)))

        model = get_embedding_model()

        # Semantic search
        if model is not None and self.embeddings is not None:
            try:
                query_embedding = model.encode([query], convert_to_numpy=True)[0]

                # Calculate similarities only for filtered documents
                similarities = []
                for idx in filtered_indices:
                    sim = np.dot(self.embeddings[idx], query_embedding)
                    similarities.append((idx, sim))

                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)

                results = []
                for idx, sim in similarities[:top_k]:
                    doc = self.documents[idx].copy()
                    doc["similarity"] = float(sim)
                    results.append(doc)
                return results

            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Fallback to keyword search
        query_lower = query.lower()
        scored_docs = []

        for idx in filtered_indices:
            doc = self.documents[idx]
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            if score > 0:
                doc_copy = doc.copy()
                doc_copy["similarity"] = score
                scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_docs[:top_k]

    def get_command(self, command_name: str) -> Optional[Dict]:
        """Get detailed information about a specific SPL command."""
        return self.commands.get(command_name.lower())

    def get_function(self, function_name: str) -> Optional[Dict]:
        """Get detailed information about a specific SPL function."""
        return self.functions.get(function_name.lower())

    def answer_question(self, question: str) -> Dict:
        """
        Answer an educational question about SPL.

        Args:
            question: User's question about SPL

        Returns:
            Dict with answer, sources, and related topics
        """
        if not self.is_initialized:
            self.initialize()

        # Search for relevant content
        results = self.search(question, top_k=5)

        if not results:
            return {
                "answer": "I don't have specific information about that. Try asking about a specific SPL command (like 'stats', 'eval', 'where'), function (like 'if', 'split', 'strftime'), or concept (like 'pipelines', 'time ranges', 'lookups').",
                "sources": [],
                "related_topics": list(self.commands.keys())[:5]
            }

        # Build answer from top results
        primary = results[0]
        answer_parts = []

        if primary["type"] == "command":
            cmd = primary["data"]
            answer_parts.append(f"## The `{cmd['name']}` Command\n")
            answer_parts.append(f"**Syntax:** `{cmd.get('syntax', '')}`\n")
            answer_parts.append(f"\n{cmd.get('description', '')}\n")

            if cmd.get("examples"):
                answer_parts.append("\n**Examples:**")
                for ex in cmd["examples"][:3]:
                    answer_parts.append(f"\n```spl\n{ex}\n```")

            if cmd.get("tips"):
                answer_parts.append("\n\n**Tips:**")
                for tip in cmd["tips"]:
                    answer_parts.append(f"\n- {tip}")

        elif primary["type"] == "function":
            func = primary["data"]
            answer_parts.append(f"## The `{func['name']}()` Function\n")
            answer_parts.append(f"**Syntax:** `{func.get('syntax', '')}`\n")
            answer_parts.append(f"\n{func.get('description', '')}\n")

            if func.get("examples"):
                answer_parts.append("\n**Examples:**")
                for ex in func["examples"][:3]:
                    answer_parts.append(f"\n```spl\n{ex}\n```")

            if func.get("tips"):
                answer_parts.append("\n\n**Tips:**")
                for tip in func["tips"]:
                    answer_parts.append(f"\n- {tip}")

        elif primary["type"] == "concept":
            concept = primary["data"]
            answer_parts.append(f"## {concept['name']}\n")
            answer_parts.append(f"{concept.get('description', '')}\n")
            if concept.get("content"):
                answer_parts.append(f"\n{concept['content']}")

        elif primary["type"] == "example":
            example = primary["data"]
            answer_parts.append(f"## {example['name']}\n")
            answer_parts.append(f"{example.get('description', '')}\n")
            answer_parts.append(f"\n**Query:**\n```spl\n{example.get('query', '')}\n```")
            if example.get("variants"):
                answer_parts.append("\n\n**Variations:**")
                for var in example["variants"]:
                    answer_parts.append(f"\n```spl\n{var}\n```")
            answer_parts.append(f"\n\n**Use case:** {example.get('use_case', '')}")

        elif primary["type"] == "best_practice":
            practice = primary["data"]
            answer_parts.append(f"## {practice['title']}\n")
            answer_parts.append(f"{practice.get('description', '')}\n")
            if practice.get("good"):
                answer_parts.append(f"\n**Good:**\n```spl\n{practice['good']}\n```")
            if practice.get("bad"):
                answer_parts.append(f"\n**Avoid:**\n```spl\n{practice['bad']}\n```")
            if practice.get("impact"):
                answer_parts.append(f"\n\n**Impact:** {practice['impact']}")

        # Collect related topics
        related = set()
        for result in results:
            if result["type"] == "command" and result.get("data", {}).get("related"):
                related.update(result["data"]["related"])
            elif result["type"] == "function" and result.get("data", {}).get("related"):
                related.update(result["data"]["related"])

        # Build sources list
        sources = [{"type": r["type"], "name": r["name"]} for r in results[:3]]

        return {
            "answer": "".join(answer_parts),
            "sources": sources,
            "related_topics": list(related)[:5],
            "similarity_score": results[0].get("similarity", 0)
        }

    def list_commands(self, category: Optional[str] = None) -> List[str]:
        """List available SPL commands, optionally filtered by category."""
        if category:
            return [name for name, data in self.commands.items() if data.get("category") == category]
        return list(self.commands.keys())

    def list_functions(self, category: Optional[str] = None) -> List[str]:
        """List available SPL functions, optionally filtered by category."""
        if category:
            return [name for name, data in self.functions.items() if data.get("category") == category]
        return list(self.functions.keys())

    def get_command_categories(self) -> List[str]:
        """Get list of command categories."""
        return list(set(data.get("category", "other") for data in self.commands.values()))

    def get_function_categories(self) -> List[str]:
        """Get list of function categories."""
        return list(set(data.get("category", "other") for data in self.functions.values()))


# Singleton instance
_spl_rag_instance = None


def get_spl_rag() -> SPLRAG:
    """Get the singleton SPLRAG instance."""
    global _spl_rag_instance
    if _spl_rag_instance is None:
        _spl_rag_instance = SPLRAG()
    return _spl_rag_instance
