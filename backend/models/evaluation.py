from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EvaluationQuery(BaseModel):
    question: str = Field(..., description="Question to evaluate")
    answer: str = Field(..., description="System's response")
    contexts: List[str] = Field(..., description="Retrieved contexts")
    ground_truth: Optional[str] = Field(None, description="Ground truth answer")
    metrics: Optional[List[str]] = Field(None, description="List of evaluation metrics")

class BatchEvaluationRequest(BaseModel):
    questions: List[str] = Field(..., description="List of questions")
    answers: List[str] = Field(..., description="List of answers")
    contexts: List[List[str]] = Field(..., description="List of contexts for each question")
    ground_truths: Optional[List[str]] = Field(None, description="List of ground truth answers")
    metrics: Optional[List[str]] = Field(None, description="List of evaluation metrics")

class SearchEvaluationQuery(BaseModel):
    query: str = Field(..., description="Search query")
    ground_truth_answer: Optional[str] = Field(None, description="Ground truth answer")
    expected_contexts: Optional[List[str]] = Field(None, description="Expected contexts")
    collection_name: str = Field("default", description="Collection name")
    top_k: int = Field(10, description="Number of results to return")
    dense_weight: float = Field(0.7, description="Weight for dense search")
    sparse_weight: float = Field(0.3, description="Weight for sparse search")

class SearchQualityTest(BaseModel):
    query: str = Field(..., description="Test query")
    ground_truth: str = Field(..., description="Ground truth answer")
    expected_contexts: Optional[List[str]] = Field(None, description="Expected contexts")

class SearchQualityEvaluation(BaseModel):
    test_queries: List[SearchQualityTest] = Field(..., description="List of test queries")
    collection_name: str = Field("default", description="Collection name")
    top_k: int = Field(10, description="Number of results to return")

class SearchBenchmark(BaseModel):
    test_queries: List[str] = Field(..., description="List of test queries")
    collection_name: str = Field("default", description="Collection name")
    top_k: int = Field(10, description="Number of results to return")
    iterations: int = Field(3, description="Number of iterations per configuration")
