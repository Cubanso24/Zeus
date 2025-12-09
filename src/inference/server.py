"""
FastAPI server for Splunk Query LLM inference.
"""

import os
import json
import psutil
import shutil
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import uvicorn
import click
from loguru import logger
import io

from src.inference.model import SplunkQueryGenerator
from src.inference.semantic_cache import (
    SemanticCache,
    initialize_semantic_cache,
    get_semantic_cache,
)
from src.inference.wazuh_rag import get_wazuh_context, fix_wazuh_query
from src.inference.splunk_client import get_splunk_client, initialize_splunk_client
from src.database.database import get_db, SessionLocal
from src.database.models import User as DBUser, Query as DBQuery, Feedback as DBFeedback, TrainingJob as DBTrainingJob
from src.database.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token,
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query generation."""
    instruction: str = Field(..., description="Natural language description of the desired Splunk query")
    input: str = Field(default="", description="Additional context or constraints")
    max_new_tokens: int = Field(default=512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    num_return_sequences: int = Field(default=1, ge=1, le=5, description="Number of queries to generate")
    explanation_format: str = Field(default="structured", description="Explanation format: 'structured' or 'paragraph'")
    indexes: Optional[List[str]] = Field(default=None, description="Splunk indexes to focus on (e.g., ['wazuh-alerts', 'main'])")


class QueryResponse(BaseModel):
    """Response model for query generation."""
    query: str = Field(..., description="Generated Splunk query or clarification request")
    explanation: Optional[str] = Field(None, description="Explanation of the generated query")
    is_clarification: bool = Field(..., description="Whether the response is a clarification request")
    clarification_questions: List[str] = Field(default=[], description="Extracted clarification questions if applicable")
    alternatives: List[str] = Field(default=[], description="Alternative query suggestions")
    query_id: Optional[int] = Field(None, description="Database ID of the saved query")
    from_cache: bool = Field(default=False, description="Whether the result came from the approved query cache")
    cache_similarity: Optional[float] = Field(None, description="Similarity score if from cache")
    cache_note: Optional[str] = Field(None, description="Note about the cached result")
    cache_params_modified: bool = Field(default=False, description="Whether parameters were adjusted for the cached query")
    cache_modifications: Optional[List[str]] = Field(None, description="List of parameter modifications made")
    # Data availability warnings
    data_warning: Optional[str] = Field(None, description="Warning about data availability limitations")
    data_warnings: List[dict] = Field(default=[], description="Detailed data availability warnings")


class BatchQueryRequest(BaseModel):
    """Request model for batch query generation."""
    requests: List[QueryRequest] = Field(..., description="List of query requests")


class BatchQueryResponse(BaseModel):
    """Response model for batch query generation."""
    responses: List[QueryResponse] = Field(..., description="List of query responses")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str


class FeedbackRequest(BaseModel):
    """User feedback on generated query."""
    query_id: Optional[int] = Field(None, description="Database ID of the query (if available)")
    instruction: str = Field(..., description="Original user instruction")
    generated_query: str = Field(..., description="Query that was generated")
    rating: str = Field(..., description="User rating: 'good' or 'bad'")
    corrected_query: Optional[str] = Field(None, description="User's corrected query if rating is bad")
    comment: Optional[str] = Field(None, description="Optional user comment")


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    status: str
    message: str


class FeedbackStats(BaseModel):
    """Feedback statistics."""
    total_feedback: int
    good_count: int
    bad_count: int
    corrections_count: int
    approval_rate: float
    recent_feedback: List[dict]


class QueryHistoryItem(BaseModel):
    """Single query history item."""
    id: int
    instruction: str
    generated_query: str
    is_clarification: bool
    temperature: float
    created_at: datetime
    has_feedback: bool = False
    feedback_rating: Optional[str] = None

    class Config:
        from_attributes = True


class QueryHistoryResponse(BaseModel):
    """Query history response."""
    queries: List[QueryHistoryItem]
    total: int
    page: int
    page_size: int


# Authentication models
class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    """User login request."""
    username: str
    password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    """User information response."""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Admin Dashboard Models
class AnalyticsResponse(BaseModel):
    """Admin analytics response."""
    total_queries: int
    total_users: int
    active_users_7d: int
    total_feedback: int
    approval_rate: float
    good_count: int
    bad_count: int
    avg_queries_per_user: float
    queries_today: int
    queries_7d: int
    queries_30d: int


class AdminQueryItem(BaseModel):
    """Admin query list item."""
    id: int
    user_id: int
    username: str
    instruction: str
    generated_query: str
    is_clarification: bool
    temperature: float
    created_at: datetime
    has_feedback: bool
    feedback_rating: Optional[str]

    class Config:
        from_attributes = True


class AdminQueryListResponse(BaseModel):
    """Admin query list response."""
    queries: List[AdminQueryItem]
    total: int
    page: int
    page_size: int


class AdminFeedbackItem(BaseModel):
    """Admin feedback list item."""
    id: int
    query_id: int
    user_id: int
    username: str
    instruction: str
    generated_query: str
    rating: str
    corrected_query: Optional[str]
    comment: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AdminFeedbackListResponse(BaseModel):
    """Admin feedback list response."""
    feedback: List[AdminFeedbackItem]
    total: int
    page: int
    page_size: int


class FeedbackUpdateRequest(BaseModel):
    """Admin feedback update request."""
    corrected_query: Optional[str] = None
    comment: Optional[str] = None


class GPUStats(BaseModel):
    """GPU statistics."""
    index: int
    name: str
    memory_total: int
    memory_used: int
    memory_percent: float
    utilization: float
    temperature: Optional[float] = None


class SystemStatsResponse(BaseModel):
    """System statistics response."""
    cpu_percent: float
    memory_total: int
    memory_used: int
    memory_percent: float
    disk_total: int
    disk_used: int
    disk_percent: float
    uptime_seconds: float
    gpu_available: bool = False
    gpus: List[GPUStats] = []


class SystemMetricsResponse(BaseModel):
    """System metrics response."""
    total_requests: int
    avg_latency_ms: float
    error_rate: float
    requests_per_minute: float


class TrainingJobRequest(BaseModel):
    """Training job start request."""
    config: Optional[Dict[str, Any]] = None
    use_feedback_data: bool = True
    min_rating: Optional[str] = None  # 'good' or None for all


class TrainingJobResponse(BaseModel):
    """Training job response."""
    id: int
    status: str
    config: Optional[Dict[str, Any]]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    model_output_path: Optional[str]
    metrics: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_by: int
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingJobListResponse(BaseModel):
    """Training job list response."""
    jobs: List[TrainingJobResponse]
    total: int
    page: int
    page_size: int


# Admin User Management Models
class AdminUserItem(BaseModel):
    """Admin user list item."""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_admin: bool
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    query_count: int = 0

    class Config:
        from_attributes = True


class AdminUserListResponse(BaseModel):
    """Admin user list response."""
    users: List[AdminUserItem]
    total: int
    page: int
    page_size: int


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    new_password: str = Field(..., min_length=6, description="New password (min 6 characters)")


class UserUpdateRequest(BaseModel):
    """User update request."""
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None


# Context and Splunk Integration Models
class IndexInfo(BaseModel):
    """Information about a Splunk index."""
    name: str
    event_count: int
    description: Optional[str] = None


class LogCategory(BaseModel):
    """Log category information."""
    id: str
    name: str
    description: str
    keywords: List[str]
    example_questions: List[str]


class ContextAnalysisRequest(BaseModel):
    """Request to analyze user instruction and determine context needs."""
    instruction: str = Field(..., description="User's natural language query")
    selected_indexes: Optional[List[str]] = Field(default=None, description="Already selected indexes")
    selected_category: Optional[str] = Field(default=None, description="Already selected log category")
    time_range: Optional[str] = Field(default=None, description="Selected time range")


class DataMismatchWarning(BaseModel):
    """Warning when user requests data their indexes can't provide."""
    requested_capability: str = Field(..., description="What the user is asking for")
    description: str = Field(..., description="Description of the requested data type")
    missing_data_type: str = Field(..., description="What data capability is missing")
    suggestion: str = Field(..., description="What to do instead or where to get this data")
    severity: str = Field(default="warning", description="warning or error")


class ContextAnalysisResponse(BaseModel):
    """Response from context analysis - determines if we need more info."""
    needs_clarification: bool = Field(..., description="Whether clarifying questions are needed")
    clarification_questions: List[str] = Field(default=[], description="Questions to ask the user")
    detected_category: Optional[str] = Field(None, description="Detected log category from instruction")
    confidence_score: float = Field(default=0.0, description="Confidence in understanding the request")
    suggested_indexes: List[str] = Field(default=[], description="Suggested indexes based on instruction")
    context_hints: List[str] = Field(default=[], description="Hints about what context would help")
    # Data capability warnings
    data_warnings: List[DataMismatchWarning] = Field(default=[], description="Warnings about data availability")
    data_source_info: Optional[str] = Field(None, description="Explanation of what data sources provide")


class QueryContextRequest(BaseModel):
    """Enhanced query request with required context."""
    instruction: str = Field(..., description="Natural language description")
    input: str = Field(default="", description="Additional context")
    # Required context fields
    indexes: List[str] = Field(..., min_length=1, description="Selected Splunk indexes (required)")
    log_category: Optional[str] = Field(None, description="Selected log category")
    time_range: Optional[str] = Field(default="-24h", description="Time range for the query")
    # Optional parameters
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    explanation_format: str = Field(default="structured")
    # Context answers (from clarification)
    context_answers: Optional[Dict[str, str]] = Field(default=None, description="Answers to clarification questions")


class SplunkConnectionStatus(BaseModel):
    """Splunk connection status."""
    connected: bool
    message: str
    indexes_available: int = 0


# Initialize FastAPI app
app = FastAPI(
    title="Splunk Query LLM API",
    description="API for generating Splunk queries using fine-tuned LLM",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[SplunkQueryGenerator] = None
model_config = {}

# Security
security = HTTPBearer()


# Authentication dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> DBUser:
    """
    Get current user from JWT token.

    Validates the JWT token and returns the authenticated user.
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    return user


# Admin authorization dependency
async def admin_required(
    current_user: DBUser = Depends(get_current_user),
) -> DBUser:
    """
    Require admin privileges.

    Validates that the current user has admin privileges.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


@app.on_event("startup")
async def startup_event():
    """Initialize model and semantic cache on startup."""
    logger.info("Starting Splunk Query LLM API server")

    # Initialize semantic cache
    try:
        device = "cuda" if os.environ.get("DEVICE", "auto") == "cuda" else "cpu"
        similarity_threshold = float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.85"))

        cache = initialize_semantic_cache(
            similarity_threshold=similarity_threshold,
            device=device
        )

        if cache.is_available():
            # Load approved queries into cache
            db = SessionLocal()
            try:
                num_loaded = cache.load_approved_queries(db)
                logger.info(f"Semantic cache initialized with {num_loaded} approved queries")
            finally:
                db.close()
        else:
            logger.warning("Semantic cache not available - sentence-transformers may not be installed")
    except Exception as e:
        logger.error(f"Failed to initialize semantic cache: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_path=model_config.get("model_path", ""),
    )


@app.get("/cache/status")
async def cache_status():
    """Get semantic cache status for debugging."""
    cache = get_semantic_cache()
    if cache is None:
        return {"available": False, "error": "Cache not initialized"}

    stats = cache.get_cache_stats()

    # Add sample of cached instructions for debugging
    sample_instructions = []
    if cache.cache:
        sample_instructions = [c.instruction[:100] for c in cache.cache[:5]]

    return {
        **stats,
        "sample_instructions": sample_instructions
    }


@app.get("/cache/test")
async def cache_test(instruction: str):
    """Test cache search for debugging."""
    from src.inference.semantic_cache import extract_time_param

    cache = get_semantic_cache()
    if cache is None:
        return {"error": "Cache not initialized"}

    if not cache.is_available():
        return {"error": "Cache not available (sentence-transformers not loaded)"}

    # Test the search
    result = cache.search(instruction)

    # Also test time extraction
    user_time = extract_time_param(instruction)

    return {
        "instruction": instruction,
        "extracted_time": user_time,
        "cache_hit": result.found,
        "similarity_score": result.similarity_score,
        "matched_instruction": result.original_instruction if result.found else None,
        "returned_query": result.query[:200] if result.query else None,
        "params_modified": result.params_modified,
        "modifications": result.modifications,
    }


# Authentication endpoints
@app.post("/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user.

    Creates a new user account and returns an access token.
    """
    # Check if username already exists
    existing_user = db.query(DBUser).filter(DBUser.username == request.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email already exists
    existing_email = db.query(DBUser).filter(DBUser.email == request.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = get_password_hash(request.password)
    new_user = DBUser(
        username=request.username,
        email=request.email,
        hashed_password=hashed_password,
        full_name=request.full_name,
        is_active=True,
        is_admin=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"New user registered: {new_user.username}")

    # Create access token
    access_token = create_access_token(data={"sub": new_user.username})

    return Token(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "full_name": new_user.full_name,
            "is_admin": new_user.is_admin,
        },
    )


@app.post("/auth/login", response_model=Token)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login with username and password.

    Validates credentials and returns an access token.
    """
    # Find user
    user = db.query(DBUser).filter(DBUser.username == request.username).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User does not exist",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    # Create access token
    access_token = create_access_token(data={"sub": user.username})

    logger.info(f"User logged in: {user.username}")

    return Token(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
        },
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: DBUser = Depends(get_current_user)):
    """
    Get current user information.

    Returns information about the currently authenticated user.
    """
    return UserResponse.from_orm(current_user)


@app.post("/generate", response_model=QueryResponse)
async def generate_query(
    request: QueryRequest,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate a Splunk query from natural language instruction.

    First checks the semantic cache for similar approved queries.
    If no match found, generates a new query using the LLM.
    Requires authentication and saves the query to the database.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Check semantic cache first for similar approved queries
        cache = get_semantic_cache()
        cache_match = None
        if cache and cache.is_available():
            logger.info(f"Searching cache for: {request.instruction}")
            cache_match = cache.search(request.instruction)
            logger.info(f"Cache search result: found={cache_match.found}, score={cache_match.similarity_score:.3f}")
            if cache_match.found:
                logger.info(f"Cache hit! Returning cached query instead of generating new one")

                # Generate explanation for cached query
                explanation = None
                try:
                    explanation = model.generate_explanation(
                        query=cache_match.query,
                        instruction=request.instruction,
                        explanation_format=request.explanation_format,
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate explanation for cached query: {e}")

                # Build cache note
                cache_note = "This query was previously approved by an analyst"
                if cache_match.was_corrected:
                    cache_note += " (with corrections)"
                cache_note += f" - {cache_match.similarity_score*100:.0f}% match"
                if cache_match.params_modified and cache_match.modifications:
                    cache_note += ". Parameters adjusted: " + ", ".join(cache_match.modifications)

                # Save to query history (mark as from cache)
                db_query = DBQuery(
                    user_id=current_user.id,
                    instruction=request.instruction,
                    input_text=request.input,
                    generated_query=cache_match.query,
                    is_clarification=False,
                    alternatives=None,
                    temperature=request.temperature,
                )
                db.add(db_query)
                db.commit()
                db.refresh(db_query)

                return QueryResponse(
                    query=cache_match.query,
                    explanation=explanation,
                    is_clarification=False,
                    clarification_questions=[],
                    alternatives=[],
                    query_id=db_query.id,
                    from_cache=True,
                    cache_similarity=round(cache_match.similarity_score, 3),
                    cache_note=cache_note,
                    cache_params_modified=cache_match.params_modified,
                    cache_modifications=cache_match.modifications,
                )

        # No cache hit - generate new query
        # Build system prompt based on user-specified indexes or defaults
        if request.indexes and len(request.indexes) > 0:
            # User specified indexes to focus on
            index_list = "\n".join([f"- {idx}" for idx in request.indexes])
            system_prompt = f"""You are a Splunk query generator. Generate valid SPL (Search Processing Language) queries.

IMPORTANT: Focus on these specific Splunk indexes:
{index_list}

When generating queries, prefer using these indexes. You may combine multiple indexes using OR if appropriate.
Use sourcetype to filter specific log types within these indexes."""

            # Add Wazuh RAG context if querying Wazuh index
            wazuh_context = get_wazuh_context(request.instruction, request.indexes)
            if wazuh_context:
                system_prompt += f"\n\n{wazuh_context}"
                logger.info("Added Wazuh RAG context to system prompt")
        else:
            # Default system prompt
            system_prompt = """You are a Splunk query generator. Generate valid SPL (Search Processing Language) queries.

IMPORTANT: Only use these standard Splunk indexes:
- main (default index for most data)
- _internal (Splunk internal logs)
- _audit (Splunk audit logs)
- os (operating system logs including Linux and Windows)
- wineventlog (Windows Event Logs)
- syslog (syslog data)

Do NOT use custom indexes like 'linux', 'firewall', 'network', 'windows', 'security', etc.
Use 'main' or 'os' for general system logs, and use sourcetype to filter specific log types."""

        # Generate query
        queries = model.generate(
            instruction=request.instruction,
            input_text=request.input,
            system_prompt=system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.num_return_sequences,
        )

        # Primary query
        primary_query = queries[0]

        # Post-process to fix Wazuh field names if applicable
        primary_query = fix_wazuh_query(primary_query, request.indexes)
        logger.info(f"Post-processed primary query: {primary_query[:100]}...")

        # Check if it's a clarification request
        is_clarification = model.is_clarification_request(primary_query)

        # Extract clarification questions if applicable
        clarification_questions = []
        explanation = None
        if is_clarification:
            clarification_questions = model.extract_clarification_questions(primary_query)
        else:
            # Generate explanation for the query
            try:
                explanation = model.generate_explanation(
                    query=primary_query,
                    instruction=request.instruction,
                    explanation_format=request.explanation_format,
                )
                if explanation:
                    logger.info(f"Explanation generated: {explanation[:100]}...")
                else:
                    logger.warning("Explanation generation returned empty result")
            except Exception as e:
                logger.warning(f"Failed to generate explanation: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                explanation = None

        # Alternative queries (also post-process for Wazuh)
        alternatives = [fix_wazuh_query(q, request.indexes) for q in queries[1:]] if len(queries) > 1 else []

        # Save query to database
        db_query = DBQuery(
            user_id=current_user.id,
            instruction=request.instruction,
            input_text=request.input,
            generated_query=primary_query,
            is_clarification=is_clarification,
            alternatives=json.dumps(alternatives) if alternatives else None,
            temperature=request.temperature,
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)

        logger.info(f"Query generated for user {current_user.username} (ID: {db_query.id})")

        return QueryResponse(
            query=primary_query,
            explanation=explanation,
            is_clarification=is_clarification,
            clarification_questions=clarification_questions,
            alternatives=alternatives,
            query_id=db_query.id,
        )

    except Exception as e:
        logger.error(f"Error generating query: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating query: {str(e)}")


@app.post("/batch_generate", response_model=BatchQueryResponse)
async def batch_generate_queries(request: BatchQueryRequest):
    """
    Generate multiple Splunk queries from a batch of instructions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        responses = []

        for req in request.requests:
            # Generate for each request
            queries = model.generate(
                instruction=req.instruction,
                input_text=req.input,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                num_return_sequences=req.num_return_sequences,
            )

            primary_query = queries[0]
            is_clarification = model.is_clarification_request(primary_query)
            clarification_questions = []

            if is_clarification:
                clarification_questions = model.extract_clarification_questions(primary_query)

            alternatives = queries[1:] if len(queries) > 1 else []

            responses.append(
                QueryResponse(
                    query=primary_query,
                    is_clarification=is_clarification,
                    clarification_questions=clarification_questions,
                    alternatives=alternatives,
                )
            )

        return BatchQueryResponse(responses=responses)

    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error in batch generation: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Collect user feedback on generated queries for model improvement.

    Saves feedback to database and also to data/feedback/user_feedback.jsonl for later retraining.
    Requires authentication.
    """
    try:
        # Find the query in the database
        query_record = None
        if feedback.query_id:
            # If query_id is provided, use it directly
            query_record = db.query(DBQuery).filter(
                DBQuery.id == feedback.query_id,
                DBQuery.user_id == current_user.id
            ).first()
        else:
            # Otherwise, try to find the most recent matching query
            query_record = db.query(DBQuery).filter(
                DBQuery.user_id == current_user.id,
                DBQuery.instruction == feedback.instruction,
                DBQuery.generated_query == feedback.generated_query
            ).order_by(DBQuery.created_at.desc()).first()

        if not query_record:
            logger.warning(f"Query not found for feedback from user {current_user.username}")
            # Continue anyway - save to file for backward compatibility

        # Save feedback to database if query found
        if query_record:
            # Check if feedback already exists for this query
            existing_feedback = db.query(DBFeedback).filter(
                DBFeedback.query_id == query_record.id
            ).first()

            if existing_feedback:
                # Update existing feedback
                existing_feedback.rating = feedback.rating
                existing_feedback.corrected_query = feedback.corrected_query
                existing_feedback.comment = feedback.comment
                existing_feedback.updated_at = datetime.utcnow()
            else:
                # Create new feedback
                db_feedback = DBFeedback(
                    query_id=query_record.id,
                    user_id=current_user.id,
                    rating=feedback.rating,
                    corrected_query=feedback.corrected_query,
                    comment=feedback.comment,
                )
                db.add(db_feedback)

            db.commit()
            logger.info(f"Feedback saved to database for user {current_user.username}, query ID {query_record.id}")

        # Also save to file for backward compatibility
        feedback_dir = "data/feedback"
        os.makedirs(feedback_dir, exist_ok=True)

        feedback_file = os.path.join(feedback_dir, "user_feedback.jsonl")

        # Prepare feedback entry
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": current_user.id,
            "username": current_user.username,
            "query_id": query_record.id if query_record else None,
            "instruction": feedback.instruction,
            "input": "",  # Keep for compatibility with training format
            "generated_output": feedback.generated_query,
            "rating": feedback.rating,
            "corrected_output": feedback.corrected_query,
            "comment": feedback.comment,
        }

        # Append to feedback file
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')

        # If rating is good, also save to approved training data and add to semantic cache
        if feedback.rating == "good":
            approved_file = os.path.join(feedback_dir, "approved_training.jsonl")
            training_entry = {
                "instruction": feedback.instruction,
                "input": "",
                "output": feedback.generated_query
            }
            with open(approved_file, 'a') as f:
                f.write(json.dumps(training_entry) + '\n')

            # Add to semantic cache for future lookups
            cache = get_semantic_cache()
            if cache and cache.is_available() and query_record:
                cache.add_to_cache(
                    query_id=query_record.id,
                    instruction=feedback.instruction,
                    generated_query=feedback.generated_query,
                    user_id=current_user.id,
                    corrected_query=feedback.corrected_query
                )
                logger.info(f"Added query {query_record.id} to semantic cache")

        # If rating is bad, handle cache accordingly
        if feedback.rating == "bad" and query_record:
            cache = get_semantic_cache()
            if cache and cache.is_available():
                if feedback.corrected_query:
                    # User provided a correction - add the CORRECTED query to cache
                    # This way, similar future questions will get the correct answer
                    cache.add_to_cache(
                        query_id=query_record.id,
                        instruction=feedback.instruction,
                        generated_query=feedback.corrected_query,  # Use corrected as the "generated"
                        user_id=current_user.id,
                        corrected_query=feedback.corrected_query
                    )
                    logger.info(f"Added corrected query {query_record.id} to semantic cache")
                else:
                    # No correction provided - just remove the bad query from cache
                    cache.remove_from_cache(query_record.id)
                    logger.info(f"Removed bad query {query_record.id} from semantic cache")

        # If rating is bad and correction provided, save to corrections training file
        if feedback.rating == "bad" and feedback.corrected_query:
            corrections_file = os.path.join(feedback_dir, "corrections_training.jsonl")
            training_entry = {
                "instruction": feedback.instruction,
                "input": "",
                "output": feedback.corrected_query
            }
            with open(corrections_file, 'a') as f:
                f.write(json.dumps(training_entry) + '\n')

        logger.info(f"Feedback received: {feedback.rating} for instruction: {feedback.instruction[:50]}...")

        return FeedbackResponse(
            status="success",
            message="Thank you for your feedback! This will help improve the model."
        )

    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


@app.get("/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats():
    """
    Get feedback statistics for monitoring model performance.

    Returns counts of good/bad ratings and recent feedback.
    """
    try:
        feedback_dir = "data/feedback"
        feedback_file = os.path.join(feedback_dir, "user_feedback.jsonl")

        # Initialize stats
        total_feedback = 0
        good_count = 0
        bad_count = 0
        corrections_count = 0
        recent_feedback = []

        # Read feedback file if it exists
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        total_feedback += 1

                        if entry.get('rating') == 'good':
                            good_count += 1
                        elif entry.get('rating') == 'bad':
                            bad_count += 1
                            if entry.get('corrected_output'):
                                corrections_count += 1

                        # Keep last 10 for recent feedback
                        recent_feedback.append({
                            'timestamp': entry.get('timestamp', ''),
                            'instruction': entry.get('instruction', '')[:60] + '...' if len(entry.get('instruction', '')) > 60 else entry.get('instruction', ''),
                            'rating': entry.get('rating', '')
                        })
                    except json.JSONDecodeError:
                        continue

        # Keep only last 10 recent
        recent_feedback = recent_feedback[-10:][::-1]  # Reverse to show newest first

        # Calculate approval rate
        approval_rate = (good_count / total_feedback * 100) if total_feedback > 0 else 0.0

        return FeedbackStats(
            total_feedback=total_feedback,
            good_count=good_count,
            bad_count=bad_count,
            corrections_count=corrections_count,
            approval_rate=round(approval_rate, 1),
            recent_feedback=recent_feedback
        )

    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        # Return empty stats on error
        return FeedbackStats(
            total_feedback=0,
            good_count=0,
            bad_count=0,
            corrections_count=0,
            approval_rate=0.0,
            recent_feedback=[]
        )


@app.get("/queries/history", response_model=QueryHistoryResponse)
async def get_query_history(
    page: int = 1,
    page_size: int = 50,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get user's query history from database with pagination.

    Returns the user's past queries ordered by most recent first.
    """
    try:
        # Calculate offset
        offset = (page - 1) * page_size

        # Get total count
        total = db.query(DBQuery).filter(DBQuery.user_id == current_user.id).count()

        # Get paginated queries with feedback information
        queries = db.query(DBQuery).filter(
            DBQuery.user_id == current_user.id
        ).order_by(DBQuery.created_at.desc()).offset(offset).limit(page_size).all()

        # Build response with feedback information
        query_items = []
        for query in queries:
            # Check if query has feedback
            feedback = db.query(DBFeedback).filter(DBFeedback.query_id == query.id).first()

            query_items.append(QueryHistoryItem(
                id=query.id,
                instruction=query.instruction,
                generated_query=query.generated_query,
                is_clarification=query.is_clarification,
                temperature=query.temperature,
                created_at=query.created_at,
                has_feedback=feedback is not None,
                feedback_rating=feedback.rating if feedback else None,
            ))

        logger.info(f"Retrieved {len(query_items)} queries for user {current_user.username} (page {page})")

        return QueryHistoryResponse(
            queries=query_items,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Error retrieving query history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving query history: {str(e)}")


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.get("/admin/analytics", response_model=AnalyticsResponse)
async def get_admin_analytics(
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Get overall analytics statistics for the admin dashboard.

    Requires admin privileges.
    """
    try:
        # Calculate time boundaries
        now = datetime.utcnow()
        seven_days_ago = now - timedelta(days=7)
        thirty_days_ago = now - timedelta(days=30)
        today_start = datetime(now.year, now.month, now.day)

        # Total queries
        total_queries = db.query(func.count(DBQuery.id)).scalar()

        # Total users
        total_users = db.query(func.count(DBUser.id)).scalar()

        # Active users in last 7 days (users who created queries)
        active_users_7d = db.query(func.count(func.distinct(DBQuery.user_id))).filter(
            DBQuery.created_at >= seven_days_ago
        ).scalar()

        # Total feedback
        total_feedback = db.query(func.count(DBFeedback.id)).scalar()

        # Good and bad counts
        good_count = db.query(func.count(DBFeedback.id)).filter(
            DBFeedback.rating == "good"
        ).scalar()
        bad_count = db.query(func.count(DBFeedback.id)).filter(
            DBFeedback.rating == "bad"
        ).scalar()

        # Approval rate
        approval_rate = (good_count / total_feedback * 100) if total_feedback > 0 else 0.0

        # Average queries per user
        avg_queries_per_user = (total_queries / total_users) if total_users > 0 else 0.0

        # Queries by time period
        queries_today = db.query(func.count(DBQuery.id)).filter(
            DBQuery.created_at >= today_start
        ).scalar()
        queries_7d = db.query(func.count(DBQuery.id)).filter(
            DBQuery.created_at >= seven_days_ago
        ).scalar()
        queries_30d = db.query(func.count(DBQuery.id)).filter(
            DBQuery.created_at >= thirty_days_ago
        ).scalar()

        logger.info(f"Admin analytics retrieved by {admin_user.username}")

        return AnalyticsResponse(
            total_queries=total_queries or 0,
            total_users=total_users or 0,
            active_users_7d=active_users_7d or 0,
            total_feedback=total_feedback or 0,
            approval_rate=round(approval_rate, 1),
            good_count=good_count or 0,
            bad_count=bad_count or 0,
            avg_queries_per_user=round(avg_queries_per_user, 1),
            queries_today=queries_today or 0,
            queries_7d=queries_7d or 0,
            queries_30d=queries_30d or 0,
        )

    except Exception as e:
        logger.error(f"Error getting admin analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")


@app.get("/admin/queries", response_model=AdminQueryListResponse)
async def get_admin_queries(
    page: int = 1,
    page_size: int = 50,
    user_id: Optional[int] = None,
    rating: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Get all queries with filters for admin dashboard.

    Requires admin privileges.
    """
    try:
        # Build query
        query = db.query(DBQuery).join(DBUser)

        # Apply filters
        if user_id:
            query = query.filter(DBQuery.user_id == user_id)

        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            query = query.filter(DBQuery.created_at >= start_dt)

        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query = query.filter(DBQuery.created_at <= end_dt)

        # Filter by rating if specified
        if rating:
            query = query.join(DBFeedback).filter(DBFeedback.rating == rating)

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * page_size
        queries = query.order_by(desc(DBQuery.created_at)).offset(offset).limit(page_size).all()

        # Build response
        query_items = []
        for q in queries:
            user = db.query(DBUser).filter(DBUser.id == q.user_id).first()
            feedback = db.query(DBFeedback).filter(DBFeedback.query_id == q.id).first()

            query_items.append(AdminQueryItem(
                id=q.id,
                user_id=q.user_id,
                username=user.username if user else "Unknown",
                instruction=q.instruction,
                generated_query=q.generated_query,
                is_clarification=q.is_clarification,
                temperature=q.temperature,
                created_at=q.created_at,
                has_feedback=feedback is not None,
                feedback_rating=feedback.rating if feedback else None,
            ))

        logger.info(f"Admin retrieved {len(query_items)} queries (page {page})")

        return AdminQueryListResponse(
            queries=query_items,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Error getting admin queries: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting queries: {str(e)}")


@app.get("/admin/feedback", response_model=AdminFeedbackListResponse)
async def get_admin_feedback(
    page: int = 1,
    page_size: int = 50,
    rating: Optional[str] = None,
    user_id: Optional[int] = None,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Get all feedback with filters for admin dashboard.

    Requires admin privileges.
    """
    try:
        # Build query
        query = db.query(DBFeedback).join(DBQuery).join(DBUser)

        # Apply filters
        if rating:
            query = query.filter(DBFeedback.rating == rating)

        if user_id:
            query = query.filter(DBFeedback.user_id == user_id)

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * page_size
        feedback_records = query.order_by(desc(DBFeedback.created_at)).offset(offset).limit(page_size).all()

        # Build response
        feedback_items = []
        for f in feedback_records:
            user = db.query(DBUser).filter(DBUser.id == f.user_id).first()
            query_record = db.query(DBQuery).filter(DBQuery.id == f.query_id).first()

            feedback_items.append(AdminFeedbackItem(
                id=f.id,
                query_id=f.query_id,
                user_id=f.user_id,
                username=user.username if user else "Unknown",
                instruction=query_record.instruction if query_record else "",
                generated_query=query_record.generated_query if query_record else "",
                rating=f.rating,
                corrected_query=f.corrected_query,
                comment=f.comment,
                created_at=f.created_at,
                updated_at=f.updated_at,
            ))

        logger.info(f"Admin retrieved {len(feedback_items)} feedback items (page {page})")

        return AdminFeedbackListResponse(
            feedback=feedback_items,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Error getting admin feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting feedback: {str(e)}")


@app.put("/admin/feedback/{feedback_id}")
async def update_admin_feedback(
    feedback_id: int,
    request: FeedbackUpdateRequest,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Update feedback item (edit corrected_query or comment).

    Requires admin privileges.
    """
    try:
        feedback = db.query(DBFeedback).filter(DBFeedback.id == feedback_id).first()

        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")

        # Update fields
        if request.corrected_query is not None:
            feedback.corrected_query = request.corrected_query

        if request.comment is not None:
            feedback.comment = request.comment

        feedback.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(feedback)

        logger.info(f"Admin {admin_user.username} updated feedback ID {feedback_id}")

        return {"status": "success", "message": "Feedback updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating feedback: {str(e)}")


@app.delete("/admin/feedback/{feedback_id}")
async def delete_admin_feedback(
    feedback_id: int,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Delete feedback item (for poor quality feedback).

    Requires admin privileges.
    """
    try:
        feedback = db.query(DBFeedback).filter(DBFeedback.id == feedback_id).first()

        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")

        db.delete(feedback)
        db.commit()

        logger.info(f"Admin {admin_user.username} deleted feedback ID {feedback_id}")

        return {"status": "success", "message": "Feedback deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting feedback: {str(e)}")


@app.post("/admin/feedback/export")
async def export_feedback(
    rating: Optional[str] = None,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Export cleaned feedback as training data (JSONL format).

    Requires admin privileges.
    """
    try:
        # Build query
        query = db.query(DBFeedback).join(DBQuery)

        # Filter by rating if specified
        if rating:
            query = query.filter(DBFeedback.rating == rating)

        feedback_records = query.all()

        # Build JSONL content
        jsonl_lines = []
        for f in feedback_records:
            query_record = db.query(DBQuery).filter(DBQuery.id == f.query_id).first()

            if not query_record:
                continue

            # For good feedback, use generated query
            # For bad feedback with correction, use corrected query
            output = query_record.generated_query
            if f.rating == "bad" and f.corrected_query:
                output = f.corrected_query

            training_entry = {
                "instruction": query_record.instruction,
                "input": query_record.input_text or "",
                "output": output
            }
            jsonl_lines.append(json.dumps(training_entry))

        # Create file-like object
        content = "\n".join(jsonl_lines)
        file_like = io.BytesIO(content.encode('utf-8'))

        logger.info(f"Admin {admin_user.username} exported {len(jsonl_lines)} feedback records")

        # Return as downloadable file
        return StreamingResponse(
            file_like,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f"attachment; filename=feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            }
        )

    except Exception as e:
        logger.error(f"Error exporting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting feedback: {str(e)}")


@app.get("/admin/system/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    admin_user: DBUser = Depends(admin_required),
):
    """
    Get system statistics (CPU, RAM, disk usage, GPU).

    Requires admin privileges.
    """
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_total = memory.total
        memory_used = memory.used
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_total = disk.total
        disk_used = disk.used
        disk_percent = disk.percent

        # System uptime
        boot_time = psutil.boot_time()
        uptime_seconds = datetime.now().timestamp() - boot_time

        # GPU stats
        gpu_available = False
        gpus = []
        try:
            import subprocess
            # Use nvidia-smi to get GPU stats
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_available = True
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            mem_total = int(parts[2]) * 1024 * 1024  # Convert MB to bytes
                            mem_used = int(parts[3]) * 1024 * 1024
                            mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                            gpus.append(GPUStats(
                                index=int(parts[0]),
                                name=parts[1],
                                memory_total=mem_total,
                                memory_used=mem_used,
                                memory_percent=round(mem_percent, 1),
                                utilization=float(parts[4]),
                                temperature=float(parts[5]) if parts[5] else None,
                            ))
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as gpu_err:
            logger.debug(f"GPU stats not available: {gpu_err}")

        return SystemStatsResponse(
            cpu_percent=round(cpu_percent, 1),
            memory_total=memory_total,
            memory_used=memory_used,
            memory_percent=round(memory_percent, 1),
            disk_total=disk_total,
            disk_used=disk_used,
            disk_percent=round(disk_percent, 1),
            uptime_seconds=round(uptime_seconds, 0),
            gpu_available=gpu_available,
            gpus=gpus,
        )

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")


# Metrics tracking start time
metrics_start_time = datetime.utcnow()


@app.get("/admin/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Get system metrics (request count, avg latency, error rate).

    Requires admin privileges. Note: This uses basic in-memory counters.
    For production, consider using proper monitoring tools.
    """
    try:
        # Use query count as a proxy for total requests
        total_requests = db.query(func.count(DBQuery.id)).scalar() or 0

        # Calculate requests per minute
        elapsed_time = (datetime.utcnow() - metrics_start_time).total_seconds() / 60
        requests_per_minute = (total_requests / elapsed_time) if elapsed_time > 0 else 0.0

        # For now, return placeholder values for latency and error rate
        # In production, these would be tracked by middleware or monitoring tools
        avg_latency_ms = 0.0
        error_rate = 0.0

        return SystemMetricsResponse(
            total_requests=total_requests,
            avg_latency_ms=round(avg_latency_ms, 2),
            error_rate=round(error_rate, 2),
            requests_per_minute=round(requests_per_minute, 2),
        )

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system metrics: {str(e)}")


@app.post("/admin/training/start", response_model=TrainingJobResponse)
async def start_training_job(
    request: TrainingJobRequest,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Trigger new model training with feedback data.

    Creates a training job record. Actual training would be handled by a background worker.
    Requires admin privileges.
    """
    try:
        # Create training job record
        training_job = DBTrainingJob(
            status="pending",
            config=request.config,
            created_by=admin_user.id,
        )

        db.add(training_job)
        db.commit()
        db.refresh(training_job)

        logger.info(f"Training job {training_job.id} created by admin {admin_user.username}")

        # In a real implementation, you would trigger a background task here
        # For example, using Celery, RQ, or FastAPI BackgroundTasks
        # background_tasks.add_task(run_training, training_job.id)

        return TrainingJobResponse.from_orm(training_job)

    except Exception as e:
        logger.error(f"Error starting training job: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting training job: {str(e)}")


@app.get("/admin/training/jobs", response_model=TrainingJobListResponse)
async def get_training_jobs(
    page: int = 1,
    page_size: int = 20,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    List training job history.

    Requires admin privileges.
    """
    try:
        # Get total count
        total = db.query(func.count(DBTrainingJob.id)).scalar()

        # Get paginated jobs
        offset = (page - 1) * page_size
        jobs = db.query(DBTrainingJob).order_by(
            desc(DBTrainingJob.created_at)
        ).offset(offset).limit(page_size).all()

        # Convert to response models
        job_responses = [TrainingJobResponse.from_orm(job) for job in jobs]

        logger.info(f"Admin retrieved {len(job_responses)} training jobs (page {page})")

        return TrainingJobListResponse(
            jobs=job_responses,
            total=total or 0,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Error getting training jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting training jobs: {str(e)}")


@app.get("/admin/training/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: int,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Get specific training job status.

    Requires admin privileges.
    """
    try:
        job = db.query(DBTrainingJob).filter(DBTrainingJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        return TrainingJobResponse.from_orm(job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training job: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting training job: {str(e)}")


# ============================================================================
# ADMIN USER MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/admin/users", response_model=AdminUserListResponse)
async def get_admin_users(
    page: int = 1,
    page_size: int = 50,
    search: Optional[str] = None,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Get all users with their stats for admin dashboard.

    Requires admin privileges.
    """
    try:
        # Build base query
        query = db.query(DBUser)

        # Apply search filter if provided
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (DBUser.username.ilike(search_term)) |
                (DBUser.email.ilike(search_term)) |
                (DBUser.full_name.ilike(search_term))
            )

        # Get total count
        total = query.count()

        # Get paginated users
        offset = (page - 1) * page_size
        users = query.order_by(desc(DBUser.created_at)).offset(offset).limit(page_size).all()

        # Build response with query counts
        user_items = []
        for user in users:
            query_count = db.query(func.count(DBQuery.id)).filter(
                DBQuery.user_id == user.id
            ).scalar() or 0

            user_items.append(AdminUserItem(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                is_admin=user.is_admin,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login,
                query_count=query_count,
            ))

        logger.info(f"Admin {admin_user.username} retrieved {len(user_items)} users (page {page})")

        return AdminUserListResponse(
            users=user_items,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Error getting admin users: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting users: {str(e)}")


@app.post("/admin/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: int,
    request: PasswordResetRequest,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Reset a user's password.

    Requires admin privileges.
    """
    try:
        # Find user
        user = db.query(DBUser).filter(DBUser.id == user_id).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Hash new password
        user.hashed_password = get_password_hash(request.new_password)
        db.commit()

        logger.info(f"Admin {admin_user.username} reset password for user {user.username} (ID: {user_id})")

        return {"message": f"Password reset successfully for user {user.username}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting password: {str(e)}")


@app.put("/admin/users/{user_id}")
async def update_user(
    user_id: int,
    request: UserUpdateRequest,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Update user status (active/admin flags).

    Requires admin privileges.
    """
    try:
        # Find user
        user = db.query(DBUser).filter(DBUser.id == user_id).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Prevent admin from deactivating themselves
        if user.id == admin_user.id and request.is_active is False:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

        # Prevent admin from removing their own admin status
        if user.id == admin_user.id and request.is_admin is False:
            raise HTTPException(status_code=400, detail="Cannot remove your own admin privileges")

        # Update fields
        if request.is_active is not None:
            user.is_active = request.is_active
        if request.is_admin is not None:
            user.is_admin = request.is_admin

        db.commit()

        logger.info(f"Admin {admin_user.username} updated user {user.username} (ID: {user_id})")

        return {"message": f"User {user.username} updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")


@app.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    admin_user: DBUser = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """
    Delete a user account.

    Requires admin privileges.
    """
    try:
        # Find user
        user = db.query(DBUser).filter(DBUser.id == user_id).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Prevent admin from deleting themselves
        if user.id == admin_user.id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")

        username = user.username
        db.delete(user)
        db.commit()

        logger.info(f"Admin {admin_user.username} deleted user {username} (ID: {user_id})")

        return {"message": f"User {username} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")


# ============================================================================
# SPLUNK INTEGRATION & CONTEXT ENDPOINTS
# ============================================================================

@app.get("/splunk/status", response_model=SplunkConnectionStatus)
async def get_splunk_status(current_user: DBUser = Depends(get_current_user)):
    """
    Check Splunk connection status.
    """
    try:
        client = get_splunk_client()
        connected, message = client.test_connection()
        indexes = client.get_available_indexes() if connected else []

        return SplunkConnectionStatus(
            connected=connected,
            message=message,
            indexes_available=len(indexes)
        )
    except Exception as e:
        logger.error(f"Error checking Splunk status: {e}")
        return SplunkConnectionStatus(
            connected=False,
            message=f"Error: {str(e)}",
            indexes_available=0
        )


@app.get("/splunk/indexes", response_model=List[IndexInfo])
async def get_splunk_indexes(current_user: DBUser = Depends(get_current_user)):
    """
    Get available Splunk indexes.

    Returns a list of indexes the user can query against.
    """
    try:
        client = get_splunk_client()
        indexes = client.get_available_indexes()

        # Add descriptions for known indexes
        index_descriptions = {
            "wazuh-alerts": "Wazuh EDR security alerts and events",
            "main": "Default index for general data",
            "_audit": "Splunk audit logs",
            "_internal": "Splunk internal logs",
        }

        return [
            IndexInfo(
                name=idx.name,
                event_count=idx.event_count,
                description=index_descriptions.get(idx.name)
            )
            for idx in indexes
        ]
    except Exception as e:
        logger.error(f"Error getting indexes: {e}")
        # Return default indexes on error
        return [
            IndexInfo(name="wazuh-alerts", event_count=0, description="Wazuh EDR security alerts"),
            IndexInfo(name="main", event_count=0, description="Default index")
        ]


@app.get("/splunk/categories", response_model=List[LogCategory])
async def get_log_categories(current_user: DBUser = Depends(get_current_user)):
    """
    Get available log categories for context selection.

    These categories help guide users in specifying what type of logs they want to query.
    """
    try:
        client = get_splunk_client()
        categories = client.get_log_categories()

        return [
            LogCategory(
                id=cat_id,
                name=cat_id.replace("_", " ").title(),
                description=info["description"],
                keywords=info["keywords"],
                example_questions=info["example_questions"]
            )
            for cat_id, info in categories.items()
        ]
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")


@app.post("/context/analyze", response_model=ContextAnalysisResponse)
async def analyze_context(
    request: ContextAnalysisRequest,
    current_user: DBUser = Depends(get_current_user),
):
    """
    Analyze user instruction and determine if clarification is needed.

    This endpoint helps determine:
    - If the request is too vague
    - What clarifying questions to ask
    - What log category the request likely falls into
    - What indexes might be relevant

    Call this BEFORE generating a query to ensure sufficient context.
    """
    try:
        client = get_splunk_client()
        instruction = request.instruction.strip()

        # Detect category from instruction
        detected_category = client.detect_category(instruction)

        # Calculate confidence score based on instruction specificity
        confidence_score = 0.0
        instruction_lower = instruction.lower()

        # Increase confidence for specific elements
        specificity_indicators = {
            "time_range": ["hour", "day", "week", "month", "last", "recent", "today", "yesterday", "earliest", "latest"],
            "target": ["host", "agent", "user", "ip", "source", "destination", "all", "specific"],
            "action": ["find", "show", "search", "count", "list", "get", "analyze", "investigate"],
            "filter": ["where", "with", "having", "equals", "contains", "greater", "less", "between"],
        }

        for category, indicators in specificity_indicators.items():
            if any(ind in instruction_lower for ind in indicators):
                confidence_score += 0.25

        # Cap at 1.0
        confidence_score = min(confidence_score, 1.0)

        # Determine if clarification is needed
        needs_clarification = confidence_score < 0.5

        # If user already provided context, increase confidence
        if request.selected_indexes and len(request.selected_indexes) > 0:
            confidence_score += 0.2
            needs_clarification = confidence_score < 0.5

        if request.selected_category:
            confidence_score += 0.1
            needs_clarification = confidence_score < 0.5
            detected_category = request.selected_category

        if request.time_range:
            confidence_score += 0.1
            needs_clarification = confidence_score < 0.5

        # Get clarification questions if needed
        clarification_questions = []
        context_hints = []

        if needs_clarification:
            clarification_questions = client.get_clarification_questions(
                instruction,
                detected_category
            )

            # Add context hints
            if not request.selected_indexes:
                context_hints.append("Select one or more indexes to search")
            if not request.time_range:
                context_hints.append("Specify a time range (e.g., 'last 24 hours')")
            if detected_category:
                context_hints.append(f"Your query appears to be about {detected_category.replace('_', ' ')}")

        # Suggest indexes based on detected category
        suggested_indexes = []
        if detected_category:
            # For security-related queries, suggest wazuh-alerts
            security_categories = ["authentication", "malware", "network", "process", "file", "compliance"]
            if detected_category in security_categories:
                suggested_indexes = ["wazuh-alerts"]
            else:
                suggested_indexes = ["main"]

        # Check for data capability mismatches
        data_warnings = []
        data_source_info = None
        indexes_to_check = request.selected_indexes or suggested_indexes or ["wazuh-alerts"]

        if indexes_to_check:
            # Check if user is asking for data their indexes can't provide
            mismatches = client.check_data_capability(instruction, indexes_to_check)

            for mismatch in mismatches:
                data_warnings.append(DataMismatchWarning(
                    requested_capability=mismatch.requested_capability,
                    description=mismatch.requested_description,
                    missing_data_type=mismatch.missing_data_type,
                    suggestion=mismatch.suggestion,
                    severity=mismatch.severity
                ))

                # Add to context hints if there's a mismatch
                if mismatch.severity == "error":
                    context_hints.insert(0, f"Your selected logs may not have {mismatch.requested_capability} data")

            # Get data source explanation
            data_source_info = client.get_data_source_explanation(indexes_to_check)

        # If there are data warnings, set needs_clarification
        if data_warnings:
            needs_clarification = True

        return ContextAnalysisResponse(
            needs_clarification=needs_clarification,
            clarification_questions=clarification_questions,
            detected_category=detected_category,
            confidence_score=round(min(confidence_score, 1.0), 2),
            suggested_indexes=suggested_indexes,
            context_hints=context_hints,
            data_warnings=data_warnings,
            data_source_info=data_source_info
        )

    except Exception as e:
        logger.error(f"Error analyzing context: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing context: {str(e)}")


@app.post("/generate/with-context", response_model=QueryResponse)
async def generate_query_with_context(
    request: QueryContextRequest,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate a Splunk query with required context.

    This endpoint requires explicit context (indexes, category) and will
    use that context to generate better, more targeted queries.

    Unlike the basic /generate endpoint, this one:
    - Requires at least one index to be specified
    - Uses context answers to refine the query
    - Includes time range constraints automatically
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build enhanced instruction with context
        enhanced_instruction = request.instruction

        # Add context answers to instruction
        if request.context_answers:
            context_parts = []
            for question, answer in request.context_answers.items():
                if answer:
                    context_parts.append(f"{question}: {answer}")
            if context_parts:
                enhanced_instruction = f"{request.instruction}\n\nAdditional context:\n" + "\n".join(context_parts)

        # Add time range to instruction if specified
        if request.time_range and request.time_range not in enhanced_instruction.lower():
            enhanced_instruction = f"{enhanced_instruction} (time range: {request.time_range})"

        # Check cache first
        cache = get_semantic_cache()
        if cache and cache.is_available():
            cache_match = cache.search(enhanced_instruction)
            if cache_match.found:
                logger.info(f"Cache hit for context query")

                # Generate explanation for cached query
                explanation = None
                try:
                    explanation = model.generate_explanation(
                        query=cache_match.query,
                        instruction=request.instruction,
                        explanation_format=request.explanation_format,
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate explanation: {e}")

                # Save to history
                db_query = DBQuery(
                    user_id=current_user.id,
                    instruction=request.instruction,
                    input_text=request.input,
                    generated_query=cache_match.query,
                    is_clarification=False,
                    alternatives=None,
                    temperature=request.temperature,
                )
                db.add(db_query)
                db.commit()
                db.refresh(db_query)

                return QueryResponse(
                    query=cache_match.query,
                    explanation=explanation,
                    is_clarification=False,
                    clarification_questions=[],
                    alternatives=[],
                    query_id=db_query.id,
                    from_cache=True,
                    cache_similarity=round(cache_match.similarity_score, 3),
                    cache_note=f"Previously approved query - {cache_match.similarity_score*100:.0f}% match",
                    cache_params_modified=cache_match.params_modified,
                    cache_modifications=cache_match.modifications,
                )

        # Build context-aware system prompt
        index_list = "\n".join([f"- {idx}" for idx in request.indexes])
        system_prompt = f"""You are a Splunk query generator. Generate valid SPL (Search Processing Language) queries.

IMPORTANT CONTEXT:
- Target indexes: {', '.join(request.indexes)}
- Time range: {request.time_range or 'not specified - use earliest=-24h as default'}
{f'- Log category: {request.log_category}' if request.log_category else ''}

Focus on these specific indexes:
{index_list}

REQUIREMENTS:
1. Always include the index specification
2. Include time constraints (earliest/latest) in the query
3. Use appropriate field names for the selected index
4. Generate practical, executable queries"""

        # Add Wazuh RAG context if applicable
        wazuh_context = get_wazuh_context(enhanced_instruction, request.indexes)
        if wazuh_context:
            system_prompt += f"\n\n{wazuh_context}"
            logger.info("Added Wazuh RAG context to system prompt")

        # Generate query
        queries = model.generate(
            instruction=enhanced_instruction,
            input_text=request.input,
            system_prompt=system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

        primary_query = queries[0]

        # Post-process for Wazuh
        primary_query = fix_wazuh_query(primary_query, request.indexes)

        # Check if it's a clarification
        is_clarification = model.is_clarification_request(primary_query)
        clarification_questions = []
        explanation = None

        if is_clarification:
            clarification_questions = model.extract_clarification_questions(primary_query)
        else:
            try:
                explanation = model.generate_explanation(
                    query=primary_query,
                    instruction=request.instruction,
                    explanation_format=request.explanation_format,
                )
            except Exception as e:
                logger.warning(f"Failed to generate explanation: {e}")

        # Save to database
        db_query = DBQuery(
            user_id=current_user.id,
            instruction=request.instruction,
            input_text=request.input,
            generated_query=primary_query,
            is_clarification=is_clarification,
            alternatives=None,
            temperature=request.temperature,
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)

        logger.info(f"Context query generated for user {current_user.username}")

        # Check for data capability mismatches and add warnings
        data_warning = None
        data_warnings_list = []

        try:
            client = get_splunk_client()
            if client.is_configured() and request.indexes:
                mismatches = client.check_data_capability(request.instruction, request.indexes)
                if mismatches:
                    # Build a summary warning
                    warning_parts = []
                    for m in mismatches:
                        warning_parts.append(f"⚠️ {m.requested_capability}: {m.suggestion}")
                        data_warnings_list.append({
                            "capability": m.requested_capability,
                            "description": m.requested_description,
                            "suggestion": m.suggestion,
                            "severity": m.severity
                        })
                    data_warning = " | ".join(warning_parts)
                    logger.info(f"Data capability warnings for query: {len(mismatches)} warnings")
        except Exception as e:
            logger.warning(f"Failed to check data capabilities: {e}")

        return QueryResponse(
            query=primary_query,
            explanation=explanation,
            is_clarification=is_clarification,
            clarification_questions=clarification_questions,
            alternatives=[],
            query_id=db_query.id,
            data_warning=data_warning,
            data_warnings=data_warnings_list,
        )

    except Exception as e:
        logger.error(f"Error generating context query: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating query: {str(e)}")


@app.post("/splunk/validate")
async def validate_query(
    query: str,
    current_user: DBUser = Depends(get_current_user),
):
    """
    Validate a Splunk query by running it against the actual Splunk instance.

    Returns whether the query is valid and how many results it returns.
    """
    try:
        client = get_splunk_client()

        if not client.is_configured():
            return {
                "valid": None,
                "message": "Splunk not configured - cannot validate",
                "result_count": 0
            }

        is_valid, message, result_count = client.validate_query(query)

        return {
            "valid": is_valid,
            "message": message,
            "result_count": result_count
        }

    except Exception as e:
        logger.error(f"Error validating query: {e}")
        return {
            "valid": False,
            "message": f"Validation error: {str(e)}",
            "result_count": 0
        }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

@click.command()
@click.option('--model-path', required=True, help='Path to fine-tuned model')
@click.option('--base-model', default=None, help='Base model name (if using PEFT adapter)')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--device', default='auto', help='Device to run model on (cpu, cuda, auto)')
@click.option('--load-in-4bit', is_flag=True, help='Load model in 4-bit quantization')
@click.option('--load-in-8bit', is_flag=True, help='Load model in 8-bit quantization')
def main(
    model_path: str,
    base_model: Optional[str],
    host: str,
    port: int,
    workers: int,
    device: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
):
    """
    Start the Splunk Query LLM API server.
    """
    global model, model_config

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = SplunkQueryGenerator(
        model_path=model_path,
        base_model=base_model,
        device=device,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    model_config = {
        "model_path": model_path,
        "base_model": base_model,
    }

    logger.info(f"Starting server on {host}:{port}")

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
