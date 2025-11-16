"""
FastAPI server for Splunk Query LLM inference.
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import click
from loguru import logger

from src.inference.model import SplunkQueryGenerator


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query generation."""
    instruction: str = Field(..., description="Natural language description of the desired Splunk query")
    input: str = Field(default="", description="Additional context or constraints")
    max_new_tokens: int = Field(default=512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    num_return_sequences: int = Field(default=1, ge=1, le=5, description="Number of queries to generate")


class QueryResponse(BaseModel):
    """Response model for query generation."""
    query: str = Field(..., description="Generated Splunk query or clarification request")
    is_clarification: bool = Field(..., description="Whether the response is a clarification request")
    clarification_questions: List[str] = Field(default=[], description="Extracted clarification questions if applicable")
    alternatives: List[str] = Field(default=[], description="Alternative query suggestions")


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


@app.post("/generate", response_model=QueryResponse)
async def generate_query(request: QueryRequest):
    """
    Generate a Splunk query from natural language instruction.

    Returns either a Splunk query or a clarification request if more information is needed.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Generate query
        queries = model.generate(
            instruction=request.instruction,
            input_text=request.input,
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
        if is_clarification:
            clarification_questions = model.extract_clarification_questions(primary_query)

        # Alternative queries
        alternatives = queries[1:] if len(queries) > 1 else []

        return QueryResponse(
            query=primary_query,
            is_clarification=is_clarification,
            clarification_questions=clarification_questions,
            alternatives=alternatives,
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


@click.command()
@click.option('--model-path', required=True, help='Path to fine-tuned model')
@click.option('--base-model', default=None, help='Base model name (if using PEFT adapter)')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--load-in-4bit', is_flag=True, help='Load model in 4-bit quantization')
@click.option('--load-in-8bit', is_flag=True, help='Load model in 8-bit quantization')
def main(
    model_path: str,
    base_model: Optional[str],
    host: str,
    port: int,
    workers: int,
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
