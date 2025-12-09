# SPL Academy Feature

## Overview
Added educational SPL learning capabilities to Zeus, allowing analysts to learn about Splunk Processing Language instead of just generating queries.

## Components

### 1. SPL RAG System (`src/inference/spl_rag.py`)
- Knowledge base with 40+ SPL commands, 40+ eval functions
- Concepts: pipelines, time ranges, indexes, fields, subsearches, lookups, etc.
- Best practices for search optimization
- Example queries for common patterns
- Semantic search using sentence transformers (same model as Wazuh RAG)

### 2. API Endpoints
- `POST /learn` - Answer educational questions about SPL
- `GET /learn/commands` - List all SPL commands (filterable by category)
- `GET /learn/commands/{name}` - Get specific command details
- `GET /learn/functions` - List all SPL functions (filterable by category)
- `GET /learn/functions/{name}` - Get specific function details
- `GET /learn/search` - Search the knowledge base

### 3. UI Changes

#### Chat Interface (`web/chat.html`)
- Mode toggle: Query Mode / Learn Mode
- In Learn Mode:
  - Different welcome message with educational example prompts
  - Hides query-specific UI (index selector, explanation format, alternatives)
  - Sends questions to /learn endpoint
  - Displays formatted educational content

#### Dedicated Academy Page (`web/academy.html`)
- Sidebar with command/function categories
- Search box with quick topic chips
- Grid view for browsing commands/functions
- Detailed view for individual commands/functions
- Related topics navigation

## Knowledge Base Categories

### Commands
- search, filtering, aggregation, transformation
- display, ordering, limiting, visualization
- correlation, enrichment, extraction
- combining, data, metadata

### Functions
- conditional (if, case, coalesce)
- string (len, lower, upper, substr, replace, split)
- conversion (tonumber, tostring)
- math (round, floor, ceil, abs, pow, sqrt)
- time (now, relative_time, strftime, strptime)
- multivalue (mvcount, mvindex, mvjoin, mvfilter)
- informational (isnull, isnotnull, typeof)
- comparison (match, like, cidrmatch)
- cryptographic (md5, sha1, sha256)

## Usage

### Learn Mode in Chat
1. Toggle to "Learn" mode in sidebar
2. Ask questions like:
   - "What does the stats command do?"
   - "How do I use the eval function?"
   - "Best practices for search optimization"

### SPL Academy Page
1. Navigate to /academy.html
2. Browse commands/functions by category
3. Search for specific topics
4. View detailed documentation with examples
