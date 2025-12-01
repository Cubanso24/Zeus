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
from src.database.database import get_db
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


class QueryResponse(BaseModel):
    """Response model for query generation."""
    query: str = Field(..., description="Generated Splunk query or clarification request")
    explanation: Optional[str] = Field(None, description="Explanation of the generated query")
    is_clarification: bool = Field(..., description="Whether the response is a clarification request")
    clarification_questions: List[str] = Field(default=[], description="Extracted clarification questions if applicable")
    alternatives: List[str] = Field(default=[], description="Alternative query suggestions")
    query_id: Optional[int] = Field(None, description="Database ID of the saved query")


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
    """Initialize model on startup."""
    logger.info("Starting Splunk Query LLM API server")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_path=model_config.get("model_path", ""),
    )


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

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
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

    Returns either a Splunk query or a clarification request if more information is needed.
    Requires authentication and saves the query to the database.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # System prompt to constrain model to use valid Splunk indexes
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

        # Alternative queries
        alternatives = queries[1:] if len(queries) > 1 else []

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

        # If rating is good, also save to approved training data
        if feedback.rating == "good":
            approved_file = os.path.join(feedback_dir, "approved_training.jsonl")
            training_entry = {
                "instruction": feedback.instruction,
                "input": "",
                "output": feedback.generated_query
            }
            with open(approved_file, 'a') as f:
                f.write(json.dumps(training_entry) + '\n')

        # If rating is bad and correction provided, save to corrections
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

@app.get("/api/admin/analytics", response_model=AnalyticsResponse)
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


@app.get("/api/admin/queries", response_model=AdminQueryListResponse)
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


@app.get("/api/admin/feedback", response_model=AdminFeedbackListResponse)
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


@app.put("/api/admin/feedback/{feedback_id}")
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


@app.delete("/api/admin/feedback/{feedback_id}")
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


@app.post("/api/admin/feedback/export")
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


@app.get("/api/admin/system/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    admin_user: DBUser = Depends(admin_required),
):
    """
    Get system statistics (CPU, RAM, disk usage).

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

        return SystemStatsResponse(
            cpu_percent=round(cpu_percent, 1),
            memory_total=memory_total,
            memory_used=memory_used,
            memory_percent=round(memory_percent, 1),
            disk_total=disk_total,
            disk_used=disk_used,
            disk_percent=round(disk_percent, 1),
            uptime_seconds=round(uptime_seconds, 0),
        )

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")


# Metrics tracking start time
metrics_start_time = datetime.utcnow()


@app.get("/api/admin/system/metrics", response_model=SystemMetricsResponse)
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


@app.post("/api/admin/training/start", response_model=TrainingJobResponse)
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


@app.get("/api/admin/training/jobs", response_model=TrainingJobListResponse)
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


@app.get("/api/admin/training/jobs/{job_id}", response_model=TrainingJobResponse)
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
