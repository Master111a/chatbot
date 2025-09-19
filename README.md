# NWS Inhouse AI Chatbot

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)
![Next.js](https://img.shields.io/badge/Next.js-18.3.2-black.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)

Hệ thống AI Chatbot thông minh với khả năng Agentic RAG (Retrieval-Augmented Generation) đa ngôn ngữ, được xây dựng cho môi trường nội bộ NewWave Solutions.

## 🚀 Tính năng chính

- **🤖 RAG-powered AI**: Tìm kiếm và trả lời dựa trên tài liệu nội bộ
- **🌐 Đa ngôn ngữ**: Hỗ trợ tiếng Việt và tiếng Anh
- **📄 Xử lý tài liệu**: Upload và phân tích PDF, DOCX, PPTX, Excel
- **🧠 Multi-Agent System**: Sử dụng nhiều AI model chuyên biệt
- **⚡ Vector Search**: Tìm kiếm ngữ nghĩa với Milvus
- **🔒 Bảo mật**: Xác thực JWT và phân quyền người dùng
- **📊 Monitoring**: Theo dõi hiệu suất và health check
- **🐳 Containerized**: Triển khai dễ dàng với Docker Compose

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │     Backend     │    │     AI Models   │
│    (Next.js)    │◄──►│    (FastAPI)    │◄──►│ (Ollama, Gemini)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼────┐  ┌───────▼────┐  ┌───────▼────┐
        │ PostgreSQL │  │   Milvus   │  │   MinIO    │
        │  (Metadata)│  │ (Vectors)  │  │ (Storage)  │
        └────────────┘  └────────────┘  └────────────┘
```

### Thành phần chính

- **Frontend**: Next.js 19 với TypeScript, TailwindCSS
- **Backend**: FastAPI với Python 3.10+, async/await
- **AI Engine**: Ollama với các model Qwen3 (1.7B, 8B)
- **Vector DB**: Milvus cho semantic search
- **Database**: PostgreSQL cho metadata
- **Storage**: MinIO cho file storage
- **Reverse Proxy**: Nginx (optional)

## 🛠️ Cài đặt và chạy

### Yêu cầu hệ thống

- Docker & Docker Compose
- RAM: Tối thiểu 8GB (khuyến nghị 16GB+)
- Disk: 20GB+ trống
- CPU: 4+ cores

### Bước 1: Clone repository

```bash
git clone https://gitlab.newwave.vn/nws/nws_inhouse_ai_chatbot.git
cd nws_inhouse_ai_chatbot
```

### Bước 2: Cấu hình environment

```bash
# Copy file environment mẫu
cp .env.example .env

# Chỉnh sửa các biến môi trường cần thiết
nano .env
```

Các biến môi trường quan trọng:

```env
# Database
PG_USER=postgres
PG_PASSWORD=your_secure_password
PG_DB=newwave_chatbot

# MinIO Storage
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key

# API Keys
GEMINI_API_KEY=your_gemini_api_key
SECRET_KEY=your_jwt_secret_key

# Ports
BE_EXPORT_PORT=17002
FE_EXPORT_PORT=3000
DB_EXPORT_PORT=17000
```

### Bước 3: Khởi chạy hệ thống

```bash
# Khởi động tất cả services
docker-compose up -d

# Theo dõi logs
docker-compose logs -f backend
```

### Bước 4: Kiểm tra hệ thống

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:17002
- **API Documentation**: http://localhost:17002/docs
- **MinIO Console**: http://localhost:9001
- **Database**: localhost:17000

## 📚 Sử dụng

### 1. Upload tài liệu

```bash
curl -X POST "http://localhost:17002/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

### 2. Chat với AI

```bash
curl -X POST "http://localhost:17002/api/v1/chat/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "message": "Hãy tóm tắt tài liệu về chính sách công ty",
    "conversation_id": "conv_123"
  }'
```

### 3. Streaming chat

```javascript
const eventSource = new EventSource(
  'http://localhost:17002/api/v1/chat/stream',
  {
    headers: {
      'Authorization': 'Bearer YOUR_TOKEN'
    }
  }
);

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log(data.content);
};
```

## 🔧 Cấu hình nâng cao

### AI Models

Hệ thống sử dụng các model chuyên biệt:

```env
# Model Configuration
REFLECTION_MODEL=qwen3:1.7b          # Phân tích query
SEMANTIC_ROUTER_MODEL=qwen3:1.7b     # Định tuyến ngữ nghĩa
RESPONSE_GENERATION_MODEL=qwen3:8b   # Sinh response
FUNCTION_CALLING_MODEL=qwen3:8b      # Gọi function
DEFAULT_CHAT_MODEL=qwen3:8b          # Chat mặc định
```

### Performance Tuning

```env
# Cache và optimization
TRANSFORMERS_CACHE=/app/model_cache
HF_HOME=/app/model_cache

# Database connections
PG_POOL_SIZE=20
PG_MAX_OVERFLOW=10

# Vector search
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_METRIC_TYPE=IP
```

## 🛡️ Bảo mật

### Authentication

- JWT tokens với expiration
- Role-based access control
- Password hashing với bcrypt

### API Security

```python
# Endpoint protection
@router.post("/protected-endpoint")
async def protected_endpoint(
    current_user: User = Depends(get_current_user)
):
    return {"message": "Access granted"}
```

## 📊 Monitoring và Health Check

### Health Endpoints

- **Basic**: `GET /health`
- **Detailed**: `GET /health/detailed`
- **Models**: `GET /health/models`

### Logging

```bash
# Xem logs của service cụ thể
docker-compose logs -f backend
docker-compose logs -f frontend

# Xem logs realtime
tail -f logs/backend/app.log
```

## 🧪 Testing

### Backend Tests

```bash
# Chạy trong container
docker exec -it newwave-backend pytest

# Hoặc local
cd backend
pip install -r requirements.txt
pytest tests/
```

### API Testing

```bash
# Test document upload
python test_document_upload.py

# Test query workflow
python test_corrected_workflow.py
```

## 🚀 Deployment

### Production Setup

1. **Cấu hình Production**:
```env
ENV=production
DEBUG=false
ENABLE_DOCS=false
```

2. **SSL/TLS**: Cấu hình HTTPS với Let's Encrypt

3. **Load Balancing**: Sử dụng multiple backend instances

4. **Backup Strategy**: 
   - PostgreSQL: pg_dump daily
   - MinIO: S3 sync
   - Milvus: Collection backup

### Docker Production

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with restart policies
docker-compose -f docker-compose.prod.yml up -d
```

## 🔍 Troubleshooting

### Lỗi thường gặp

1. **Model loading failed**:
```bash
# Kiểm tra Ollama status
docker exec -it ollama ollama list
docker exec -it ollama ollama pull qwen3:8b
```

2. **Database connection error**:
```bash
# Kiểm tra PostgreSQL
docker exec -it db-postgres psql -U postgres -d newwave_chatbot
```

3. **Vector search error**:
```bash
# Kiểm tra Milvus
docker exec -it db-milvus curl localhost:9091/healthz
```

## 📈 Performance Benchmarks

- **Response time**: < 2s (average)
- **Throughput**: 100+ requests/minute
- **Memory usage**: ~4GB (với models loaded)
- **Vector search**: < 100ms

## 🤝 Contributing

### Development Setup

```bash
# Backend development
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend development  
cd frontend
npm install
npm run dev
```

### Code Style

- Backend: Black, isort, flake8
- Frontend: ESLint, Prettier
- Commit messages: Conventional commits

## 📄 License

Copyright © 2024 NewWave Solutions. All rights reserved.

## 👥 Team

- **Backend**: Python/FastAPI Team
- **Frontend**: React/Next.js Team  
- **AI/ML**: Machine Learning Team
- **DevOps**: Infrastructure Team
