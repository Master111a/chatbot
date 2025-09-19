# Hướng Dẫn Sử Dụng Ragas Evaluation

## Tổng Quan

Ragas (RAG Assessment) là một framework mạnh mẽ để đánh giá chất lượng hệ thống RAG (Retrieval-Augmented Generation). Chúng tôi đã tích hợp Ragas vào HybridSearchService để cung cấp các tính năng đánh giá toàn diện.

## Cài Đặt

Ragas đã được thêm vào requirements.txt:
```
ragas==0.1.18
datasets==2.16.1
```

## Các Metrics Hỗ Trợ

### 1. Answer Relevancy
- **Mục đích**: Đánh giá độ liên quan của câu trả lời với câu hỏi
- **Điểm số**: 0-1 (càng cao càng tốt)
- **Sử dụng khi**: Có câu hỏi và câu trả lời

### 2. Answer Correctness  
- **Mục đích**: Đánh giá độ chính xác của câu trả lời so với ground truth
- **Điểm số**: 0-1 (càng cao càng tốt)
- **Sử dụng khi**: Có ground truth answer

### 3. Context Precision
- **Mục đích**: Đánh giá độ chính xác của context được retrieve
- **Điểm số**: 0-1 (càng cao càng tốt)
- **Sử dụng khi**: Đánh giá quality của retrieval

### 4. Context Recall
- **Mục đích**: Đánh giá khả năng retrieve đầy đủ context cần thiết
- **Điểm số**: 0-1 (càng cao càng tốt)
- **Sử dụng khi**: Có ground truth context

### 5. Context Relevancy
- **Mục đích**: Đánh giá độ liên quan của context với câu hỏi
- **Điểm số**: 0-1 (càng cao càng tốt)
- **Sử dụng khi**: Đánh giá quality của context

### 6. Faithfulness
- **Mục đích**: Đánh giá tính trung thực (câu trả lời có dựa trên context không)
- **Điểm số**: 0-1 (càng cao càng tốt)
- **Sử dụng khi**: Kiểm tra hallucination

### 7. Answer Similarity
- **Mục đích**: Đánh giá độ tương đồng ngữ nghĩa giữa answer và ground truth
- **Điểm số**: 0-1 (càng cao càng tốt)
- **Sử dụng khi**: So sánh semantic similarity

## Cách Sử Dụng

### 1. Đánh Giá Query Đơn Lẻ

```python
from services.ragas_evaluation_service import ragas_evaluation_service

result = await ragas_evaluation_service.evaluate_single_query(
    question="Newwave Solutions có bao nhiêu năm kinh nghiệm?",
    answer="Newwave có hơn 10 năm kinh nghiệm.",
    contexts=["Newwave thành lập 2011", "Có 10+ năm kinh nghiệm"],
    ground_truth="Newwave có hơn 10 năm kinh nghiệm trong IT."
)

print(result["metrics"])
```

### 2. Đánh Giá Batch

```python
result = await ragas_evaluation_service.evaluate_batch(
    questions=["Câu hỏi 1", "Câu hỏi 2"],
    answers=["Trả lời 1", "Trả lời 2"], 
    contexts=[["Context 1"], ["Context 2"]],
    ground_truths=["Ground truth 1", "Ground truth 2"]
)

print(result["metric_statistics"])
```

### 3. Search Với Evaluation Tự Động

```python
from services.hybrid_search_service import hybrid_search_service

# Enable evaluation tracking
hybrid_search_service.enable_evaluation(True)

result = await hybrid_search_service.search_with_evaluation(
    query="Newwave Solutions có bao nhiêu năm kinh nghiệm?",
    ground_truth_answer="Newwave có hơn 10 năm kinh nghiệm.",
    collection_name="default",
    top_k=5
)

print(result["evaluation"]["metrics"])
```

### 4. Đánh Giá Chất Lượng Search

```python
test_queries = [
    {
        "query": "Câu hỏi test 1",
        "ground_truth": "Câu trả lời chuẩn 1",
        "expected_contexts": ["Context mong đợi"]
    },
    # ... thêm queries
]

result = await hybrid_search_service.evaluate_search_quality(
    test_queries=test_queries,
    collection_name="default",
    top_k=10
)

print(result["metric_statistics"])
```

### 5. Benchmark Search Methods

```python
test_queries = ["Query 1", "Query 2", "Query 3"]

result = await hybrid_search_service.benchmark_search_methods(
    test_queries=test_queries,
    collection_name="default",
    top_k=5,
    iterations=3
)

print(f"Best config: {result['best_configuration']}")
```

## API Endpoints

### POST /api/v1/evaluation/single-query
Đánh giá một query/response pair duy nhất.

```json
{
  "question": "Câu hỏi",
  "answer": "Câu trả lời", 
  "contexts": ["Context 1", "Context 2"],
  "ground_truth": "Câu trả lời chuẩn"
}
```

### POST /api/v1/evaluation/batch
Đánh giá batch queries.

```json
{
  "questions": ["Q1", "Q2"],
  "answers": ["A1", "A2"],
  "contexts": [["C1"], ["C2"]],
  "ground_truths": ["GT1", "GT2"]
}
```

### POST /api/v1/evaluation/search-with-evaluation
Search kèm evaluation tự động.

```json
{
  "query": "Câu hỏi search",
  "ground_truth_answer": "Câu trả lời chuẩn",
  "collection_name": "default",
  "top_k": 10
}
```

### POST /api/v1/evaluation/search-quality
Đánh giá chất lượng search.

```json
{
  "test_queries": [
    {
      "query": "Query",
      "ground_truth": "Ground truth",
      "expected_contexts": ["Context"]
    }
  ],
  "collection_name": "default",
  "top_k": 10
}
```

### POST /api/v1/evaluation/benchmark
Benchmark search methods.

```json
{
  "test_queries": ["Q1", "Q2", "Q3"],
  "collection_name": "default",
  "top_k": 5,
  "iterations": 3
}
```

### GET /api/v1/evaluation/summary
Lấy tóm tắt evaluation results.

### GET /api/v1/evaluation/performance-metrics
Lấy performance metrics.

### POST /api/v1/evaluation/enable-tracking
Bật/tắt evaluation tracking.

```json
{
  "enabled": true
}
```

### GET /api/v1/evaluation/status
Kiểm tra trạng thái evaluation system.

## Ví Dụ Thực Tế

### 1. Chạy Example Script

```bash
cd backend
python examples/ragas_evaluation_example.py
```

### 2. Test API với cURL

```bash
# Kiểm tra status
curl -X GET "http://localhost:17002/api/v1/evaluation/status"

# Đánh giá single query
curl -X POST "http://localhost:17002/api/v1/evaluation/single-query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Newwave có bao nhiêu năm kinh nghiệm?",
    "answer": "Newwave có hơn 10 năm kinh nghiệm.",
    "contexts": ["Newwave thành lập 2013", "Có 10+ năm kinh nghiệm"],
    "ground_truth": "Newwave có hơn 10 năm kinh nghiệm trong IT."
  }'

# Enable evaluation tracking
curl -X POST "http://localhost:17002/api/v1/evaluation/enable-tracking?enabled=true"

# Search với evaluation
curl -X POST "http://localhost:17002/api/v1/evaluation/search-with-evaluation" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Newwave có những dịch vụ gì?",
    "ground_truth_answer": "Newwave cung cấp dịch vụ phát triển phần mềm.",
    "top_k": 5
  }'
```

## Best Practices

### 1. Chuẩn Bị Test Data
- Tạo một bộ test queries với ground truth answers
- Đảm bảo đa dạng về loại câu hỏi (factual, contextual, etc.)
- Chuẩn bị expected contexts cho mỗi query

### 2. Chọn Metrics Phù Hợp
- **Context evaluation**: context_precision, context_recall, context_relevancy
- **Answer evaluation**: answer_relevancy, answer_correctness, faithfulness
- **Semantic similarity**: answer_similarity

### 3. Interpretation Results
- **> 0.8**: Excellent quality
- **0.6 - 0.8**: Good quality
- **0.4 - 0.6**: Acceptable, có thể cải thiện
- **< 0.4**: Poor quality, cần cải thiện ngay

### 4. Monitoring và Tracking
- Enable evaluation tracking trong production
- Định kỳ chạy evaluation với new test data
- Export results để phân tích xu hướng

### 5. Performance Optimization
- Sử dụng batch evaluation cho large datasets
- Monitor evaluation time để tối ưu
- Cache evaluation results khi có thể

## Troubleshooting

### 1. Ragas Not Available
```
Error: Ragas framework not installed
```
**Giải pháp**: Cài đặt ragas package
```bash
pip install ragas==0.1.18 datasets==2.16.1
```

### 2. Memory Issues
```
Error: Out of memory during evaluation
```
**Giải pháp**: 
- Giảm batch size
- Sử dụng single query evaluation
- Tăng memory cho container

### 3. API Timeout
```
Error: Request timeout
```
**Giải pháp**:
- Tăng request timeout
- Giảm số lượng queries per request
- Sử dụng background tasks

### 4. No Evaluation Results
```
Message: No evaluation results available
```
**Giải pháp**:
- Chạy ít nhất một evaluation trước
- Enable evaluation tracking
- Kiểm tra lỗi trong logs

## Tích Hợp CI/CD

### 1. Automated Testing
```python
# tests/test_evaluation.py
async def test_search_quality():
    result = await hybrid_search_service.evaluate_search_quality(
        test_queries=TEST_QUERIES,
        collection_name="test",
        top_k=5
    )
    
    # Assert minimum quality thresholds
    for metric, stats in result["metric_statistics"].items():
        assert stats["mean"] > 0.4, f"{metric} below threshold"
```

### 2. Quality Gates
```yaml
# .github/workflows/quality-check.yml
- name: Run Evaluation Tests
  run: |
    python -m pytest tests/test_evaluation.py
    python scripts/quality_check.py
```

### 3. Performance Monitoring
```python
# scripts/quality_check.py
async def main():
    result = await hybrid_search_service.benchmark_search_methods(
        test_queries=BENCHMARK_QUERIES,
        iterations=5
    )
    
    # Check performance regression
    avg_time = result["results_by_configuration"]["hybrid_70_30"]["average_time"]
    assert avg_time < 1.0, "Performance regression detected"
```

Với hệ thống evaluation này, bạn có thể:
- Đánh giá chất lượng search results một cách khách quan
- Benchmark các configuration khác nhau
- Monitor performance theo thời gian
- Phát hiện regression trong chất lượng
- Tối ưu parameters dựa trên metrics 