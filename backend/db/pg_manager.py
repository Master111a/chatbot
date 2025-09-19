import json
import asyncpg
import uuid

from typing import Dict, List, Any, Optional
from asyncpg.pool import Pool
from utils import now

from models import (
    Document,
    DocumentChunk,
    DocumentStore
)

from utils.logger import get_logger

logger = get_logger(__name__)

class PostgresManager:
    """
    Main PostgreSQL connection manager.
    Handles database connections, schema initialization, and common utilities.
    """
    _pool: Optional[Pool] = None
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PostgresManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    async def initialize(self):
        """Initialize the database connection pool and schema"""
        if self.initialized:
            return
            
        from config.settings import get_settings
        settings = get_settings()
        
        try:
            database_url = settings.DATABASE_URL
            self._pool = await asyncpg.create_pool(
                database_url,
                min_size=5,
                max_size=25,
                command_timeout=30,
                max_inactive_connection_lifetime=300,
                max_queries=50000,
                setup=self._setup_connection
            )
            
            self.initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PostgreSQL connection: {str(e)}")
    
    async def _setup_connection(self, conn):
        """Setup individual database connections for better performance"""
        await conn.execute("SET application_name = 'nws_chatbot'")
        await conn.execute("SET timezone = 'Asia/Ho_Chi_Minh'")
        await conn.execute("SET plan_cache_mode = 'force_generic_plan'")
    
    async def close(self):
        """Close the connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self.initialized = False

    @property
    def pool(self) -> Pool:
        """Get the connection pool"""
        if not self._pool:
            raise RuntimeError("Database connection pool not initialized")
        return self._pool


class UserManager:
    """
    User management in database.
    """
    
    def __init__(self):
        self.pg_manager = PostgresManager()
    
    async def create_or_update_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update user from GitLab data.
        """
        async with self.pg_manager.pool.acquire() as conn:
            current_time = now()
            
            existing_user = await conn.fetchrow(
                'SELECT * FROM users WHERE gitlab_id = $1',
                user_data["gitlab_id"]
            )
            
            if existing_user:
                await conn.execute('''
                    UPDATE users SET
                        email = $1, username = $2, name = $3, avatar_url = $4,
                        is_superuser = $5, last_login = $6
                    WHERE gitlab_id = $7
                ''', 
                user_data["email"], user_data["username"], user_data["name"],
                user_data["avatar_url"], user_data["is_superuser"], current_time,
                user_data["gitlab_id"])
                
                user_id = existing_user["user_id"]
            else:
                user_id = str(uuid.uuid4())
                await conn.execute('''
                    INSERT INTO users (
                        user_id, email, username, name, avatar_url, 
                        gitlab_id, is_superuser, date_joined, last_login,
                        is_active, is_staff, is_superuser
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ''',
                user_id, user_data["email"], user_data["username"], user_data["name"],
                user_data["avatar_url"], user_data["gitlab_id"], user_data["is_superuser"],
                current_time, current_time, True, False, user_data.get("is_superuser", False))
            
            user = await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
            
            user_dict = dict(user)
            if user_dict.get('date_joined'):
                user_dict['date_joined'] = user_dict['date_joined'].isoformat()
            if user_dict.get('last_login'):
                user_dict['last_login'] = user_dict['last_login'].isoformat()
                
            return user_dict
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID.
        """
        async with self.pg_manager.pool.acquire() as conn:
            user = await conn.fetchrow('SELECT * FROM users WHERE user_id = $1', user_id)
            return dict(user) if user else None
        
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username.
        """
        async with self.pg_manager.pool.acquire() as conn:
            user = await conn.fetchrow('SELECT * FROM users WHERE username = $1', username)
            return dict(user) if user else None
    
    async def update_last_login(self, user_id: str) -> bool:
        """
        Update user's last login timestamp.
        """
        async with self.pg_manager.pool.acquire() as conn:
            current_time = now()
            result = await conn.execute(
                'UPDATE users SET last_login = $1 WHERE user_id = $2',
                current_time, user_id
            )
            return 'UPDATE' in result


class PostgresDocumentStore(DocumentStore):
    """PostgreSQL implementation of DocumentStore interface"""
    
    def __init__(self):
        self.pg_manager = PostgresManager()
    
    async def create_document(self, document: Document) -> str:
        """Store a new document"""
        async with self.pg_manager.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO documents (
                    document_id, title, file_name, file_type, file_size,
                    user_id, language, description, status, chunks_count,
                    chunk_size, chunk_overlap, metadata, created_at, updated_at, file_path
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            ''', document.document_id, document.title, document.file_name,
            document.file_type, document.file_size, document.user_id,
            document.language, document.description, document.status,
            document.chunks_count, document.chunk_size, document.chunk_overlap,
            json.dumps(document.metadata), document.created_at, document.updated_at,
            getattr(document, 'file_path', None))
            
            return document.document_id
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID"""
        async with self.pg_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT * FROM documents WHERE document_id = $1', 
                document_id
            )
            
            if not row:
                return None
                
            doc_dict = dict(row)
            doc_dict['metadata'] = json.loads(doc_dict['metadata'])
            
            if 'document_id' in doc_dict and doc_dict['document_id']:
                doc_dict['document_id'] = str(doc_dict['document_id'])
            if 'user_id' in doc_dict and doc_dict['user_id']:
                doc_dict['user_id'] = str(doc_dict['user_id'])
            
            return Document.from_dict(doc_dict)
    
    async def update_document(self, document: Document) -> bool:
        """Update a document"""
        async with self.pg_manager.pool.acquire() as conn:
            result = await conn.execute('''
                UPDATE documents SET
                    title = $1, file_name = $2, file_type = $3, file_size = $4,
                    user_id = $5, language = $6, description = $7, status = $8,
                    chunks_count = $9, chunk_size = $10, chunk_overlap = $11,
                    metadata = $12, updated_at = $13, file_path = $14
                WHERE document_id = $15
            ''', document.title, document.file_name, document.file_type,
            document.file_size, document.user_id, document.language,
            document.description, document.status, document.chunks_count,
            document.chunk_size, document.chunk_overlap,
            json.dumps(document.metadata), document.updated_at,
            getattr(document, 'file_path', None), document.document_id)
            
            return 'UPDATE' in result
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        async with self.pg_manager.pool.acquire() as conn:
            result = await conn.execute(
                'DELETE FROM documents WHERE document_id = $1', 
                document_id
            )
            return 'DELETE' in result
    
    async def list_documents(
        self,
        user_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[Document]:
        """List documents with pagination and filtering"""
        async with self.pg_manager.pool.acquire() as conn:
            params = []
            query = "SELECT * FROM documents"
            conditions = []
            
            if user_id:
                conditions.append(f"user_id = ${len(params) + 1}")
                params.append(user_id)
                
            if search:
                conditions.append(f"(title ILIKE ${len(params) + 1} OR description ILIKE ${len(params) + 1})")
                params.append(f"%{search}%")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
            params.extend([limit, offset])
            
            rows = await conn.fetch(query, *params)
            
            documents = []
            for row in rows:
                doc_dict = dict(row)
                doc_dict['metadata'] = json.loads(doc_dict['metadata'])
                
                if 'document_id' in doc_dict and doc_dict['document_id']:
                    doc_dict['document_id'] = str(doc_dict['document_id'])
                if 'user_id' in doc_dict and doc_dict['user_id']:
                    doc_dict['user_id'] = str(doc_dict['user_id'])
                
                documents.append(Document.from_dict(doc_dict))
                
            return documents
    
    async def add_chunk(self, chunk: DocumentChunk) -> str:
        """Add a document chunk"""
        async with self.pg_manager.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute('''
                    INSERT INTO document_chunks (
                        chunk_id, document_id, content_path,
                        metadata, index, chunk_size, created_at,
                        prev_chunk_id, next_chunk_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''', chunk.chunk_id, chunk.document_id, chunk.content_path,
                json.dumps(chunk.metadata), chunk.index,
                chunk.chunk_size, chunk.created_at,
                chunk.prev_chunk_id, chunk.next_chunk_id)
                
                await conn.execute('''
                    UPDATE documents 
                    SET chunks_count = chunks_count + 1,
                        updated_at = $1
                    WHERE document_id = $2
                ''', now().isoformat(), chunk.document_id)
                
            return chunk.chunk_id
    
    async def get_chunks(
        self, 
        document_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentChunk]:
        """Get chunks for a document with pagination"""
        async with self.pg_manager.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM document_chunks 
                WHERE document_id = $1
                ORDER BY index ASC
                LIMIT $2 OFFSET $3
            ''', document_id, limit, offset)
            
            chunks = []
            for row in rows:
                chunk_dict = dict(row)
                chunk_dict['metadata'] = json.loads(chunk_dict['metadata'])
                chunks.append(DocumentChunk.from_dict(chunk_dict))
                
            return chunks
    
    async def delete_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        async with self.pg_manager.pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    'DELETE FROM document_chunks WHERE document_id = $1', 
                    document_id
                )
                
                await conn.execute('''
                    UPDATE documents 
                    SET chunks_count = 0, updated_at = $1
                    WHERE document_id = $2
                ''', now().isoformat(), document_id)
                
            return 'DELETE' in result


class ChatHistory:
    """
    PostgreSQL-based chat history manager with user support.
    """
    
    def __init__(self):
        self.pg_manager = PostgresManager()
    
    async def add_message(
        self,
        session_id: str,
        message: str,
        response: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a message to a chat session and return the created message including its ID."""
        async with self.pg_manager.pool.acquire() as conn:
            async with conn.transaction():
                message_id = str(uuid.uuid4())
                timestamp = now().isoformat()
                
                metadata_json = json.dumps(metadata) if metadata else None
                
                await conn.execute(
                    """INSERT INTO chat_messages 
                        (message_id, session_id, user_id, message, response, metadata, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    message_id, session_id, user_id, message, response, metadata_json, timestamp
                )
                
                await conn.execute('''
                    UPDATE chat_sessions SET
                        message_count = message_count + 1,
                        updated_at = $1
                    WHERE session_id = $2
                ''', timestamp, session_id)
                
                return {
                    "message_id": message_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "message": message,
                    "response": response,
                    "metadata": metadata,
                    "timestamp": timestamp
                }

    async def update_message_metadata(self, message_id: str, metadata_update: Dict[str, Any]) -> bool:
        """
        Update the metadata of a specific message.
        Merges the new metadata_update with existing metadata.
        """
        async with self.pg_manager.pool.acquire() as conn:
            try:
                async with conn.transaction():
                    current_metadata_row = await conn.fetchrow(
                        "SELECT metadata FROM chat_messages WHERE message_id = $1",
                        message_id
                    )

                    if not current_metadata_row:
                        logger.warning(f"Message with ID {message_id} not found for metadata update.")
                        return False

                    current_metadata = current_metadata_row['metadata']
                    if isinstance(current_metadata, str):
                        try:
                            current_metadata = json.loads(current_metadata)
                        except json.JSONDecodeError:
                            logger.error(f"Could not parse existing metadata for message {message_id}. It will be overwritten.")
                            current_metadata = {}
                    elif not isinstance(current_metadata, dict):
                        logger.warning(f"Existing metadata for message {message_id} is not a dict. It will be overwritten.")
                        current_metadata = {}
                    
                    if not isinstance(metadata_update, dict):
                        logger.warning(f"metadata_update for message {message_id} is not a dict ({type(metadata_update)}). Using empty dict instead.")
                        metadata_update = {}

                    updated_metadata = {**current_metadata, **metadata_update}
                    
                    result = await conn.execute(
                        """
                        UPDATE chat_messages
                        SET metadata = $1
                        WHERE message_id = $2
                        """,
                        json.dumps(updated_metadata),
                        message_id
                    )
                    
                    logger.info(f"Metadata for message {message_id} updated. Result: {result}")
                    return 'UPDATE 1' in result or (isinstance(result, str) and result.startswith("UPDATE") and int(result.split(" ")[1]) > 0)

            except Exception as e:
                logger.error(f"Error updating metadata for message {message_id}: {e}", exc_info=True)
                return False

    async def create_session(
        self, 
        title: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new chat session with optional title.
        """
        session_id = str(uuid.uuid4())
        timestamp = now().isoformat()
        
        async with self.pg_manager.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO chat_sessions (
                    session_id, user_id, title, created_at, updated_at, 
                    message_count, metadata, is_anonymous
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', session_id, user_id, title, timestamp, timestamp, 0, '{}', user_id is None)
            
            session = await self.get_session_info(session_id)
            return session
    
    async def update_session_title(
        self,
        session_id: str,
        title: str
    ) -> bool:
        """
        Update chat session title.
        """
        async with self.pg_manager.pool.acquire() as conn:
            timestamp = now().isoformat()
            result = await conn.execute('''
                UPDATE chat_sessions SET
                    title = $1, updated_at = $2
                WHERE session_id = $3
            ''', title, timestamp, session_id)
            
            return 'UPDATE' in result
    
    async def find_similar_session(
        self,
        first_question: str,
        user_id: Optional[str] = None,
        limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Find similar chat session based on first question.
        Only returns sessions that belong to the same user for privacy.
        Anonymous users cannot access other anonymous users' sessions.
        """
        async with self.pg_manager.pool.acquire() as conn:
            query = '''
                SELECT s.*, m.message, similarity(m.message, $1) as score
                FROM chat_sessions s
                JOIN (
                    SELECT DISTINCT ON (session_id) 
                        session_id, message, timestamp
                    FROM chat_messages
                    ORDER BY session_id, timestamp ASC
                ) m ON s.session_id = m.session_id
                WHERE similarity(m.message, $1) > 0.6
            '''
            
            params = [first_question]
            
            if user_id:
                query += " AND (s.user_id = $2 OR s.is_anonymous = TRUE)"
                params.append(user_id)
            else:
                query += " AND FALSE"

            query += " ORDER BY score DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            if not rows:
                return None
                
            session_dict = dict(rows[0])
            session_dict['metadata'] = json.loads(session_dict['metadata'])
            session_dict['first_message'] = rows[0]['message']
            session_dict['similarity_score'] = rows[0]['score']
            
            return session_dict
    
    async def get_history(
        self,
        session_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a session.
        """
        async with self.pg_manager.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM chat_messages
                WHERE session_id = $1
                ORDER BY timestamp ASC
                LIMIT $2 OFFSET $3
            ''', session_id, limit, offset)
            
            messages = []
            for row in rows:
                message_dict = dict(row)
                try:
                    if message_dict['metadata']:
                        message_dict['metadata'] = json.loads(message_dict['metadata'])
                    else:
                        message_dict['metadata'] = {}
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid metadata for message {message_dict.get('message_id')}")
                    message_dict['metadata'] = {}
                    
                messages.append(message_dict)
                
            return messages
    
    async def get_recent_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent chat history for a session.
        Returns messages ordered from newest to oldest.
        """
        async with self.pg_manager.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM chat_messages
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            ''', session_id, limit)
            
            messages = []
            for row in rows:
                message_dict = dict(row)
                try:
                    if message_dict['metadata']:
                        message_dict['metadata'] = json.loads(message_dict['metadata'])
                    else:
                        message_dict['metadata'] = {}
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid metadata for message {message_dict.get('message_id')}")
                    message_dict['metadata'] = {}
                    
                messages.append(message_dict)
                
            return messages
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        include_empty: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user with filtering options.
        """
        async with self.pg_manager.pool.acquire() as conn:
            base_query = '''
                SELECT s.*, 
                       COALESCE(s.message_count, 0) as message_count,
                       (SELECT message FROM chat_messages WHERE session_id = s.session_id ORDER BY timestamp DESC LIMIT 1) as last_message,
                       COALESCE(
                           (SELECT timestamp FROM chat_messages WHERE session_id = s.session_id ORDER BY timestamp DESC LIMIT 1), 
                           s.updated_at
                       ) as last_activity,
                       (SELECT metadata->>'language' FROM chat_messages WHERE session_id = s.session_id AND metadata->>'language' IS NOT NULL ORDER BY timestamp ASC LIMIT 1) as language
                FROM chat_sessions s
                WHERE s.user_id = $1
            '''
            
            if not include_empty:
                base_query += " AND COALESCE(s.message_count, 0) > 0"
            
            base_query += '''
                ORDER BY COALESCE(
                    (SELECT timestamp FROM chat_messages WHERE session_id = s.session_id ORDER BY timestamp DESC LIMIT 1), 
                    s.updated_at
                ) DESC
                LIMIT $2 OFFSET $3
            '''
            
            rows = await conn.fetch(base_query, user_id, limit, offset)
            
            sessions = []
            for row in rows:
                session_dict = dict(row)
                try:
                    session_dict['metadata'] = json.loads(session_dict['metadata'])
                except (json.JSONDecodeError, TypeError):
                    session_dict['metadata'] = {}
                sessions.append(session_dict)
                
            return sessions
    
    async def count_user_sessions(
        self,
        user_id: str,
        include_empty: bool = False
    ) -> int:
        """
        Count total number of sessions for a user.
        """
        async with self.pg_manager.pool.acquire() as conn:
            query = '''
                SELECT COUNT(*) 
                FROM chat_sessions 
                WHERE user_id = $1
            '''
            
            if not include_empty:
                query += " AND COALESCE(message_count, 0) > 0"
            
            count = await conn.fetchval(query, user_id)
            return count or 0
    
    async def link_anonymous_session_to_user(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """
        Link anonymous session to user when they log in.
        """
        async with self.pg_manager.pool.acquire() as conn:
            async with conn.transaction():
                session_exists = await conn.fetchval(
                    'SELECT EXISTS(SELECT 1 FROM chat_sessions WHERE session_id = $1)',
                    session_id
                )
                
                if not session_exists:
                    return False
                
                current_time = now().isoformat()
                
                await conn.execute('''
                    UPDATE chat_sessions SET
                        user_id = $1, is_anonymous = FALSE, updated_at = $2
                    WHERE session_id = $3
                ''', user_id, current_time, session_id)
                
                await conn.execute('''
                    UPDATE chat_messages SET
                        user_id = $1
                    WHERE session_id = $2 AND user_id IS NULL
                ''', user_id, session_id)
                
                return True
    
    async def clear_history(self, session_id: str) -> bool:
        """
        Clear chat history for a session.
        """
        async with self.pg_manager.pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    'DELETE FROM chat_messages WHERE session_id = $1',
                    session_id
                )
                
                await conn.execute('''
                    UPDATE chat_sessions SET
                        message_count = 0, 
                        updated_at = $1
                    WHERE session_id = $2
                ''', now().isoformat(), session_id)
                
            return 'DELETE' in result
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete entire chat session and all its messages.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if session was deleted, False if not found
        """
        async with self.pg_manager.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    'DELETE FROM chat_messages WHERE session_id = $1',
                    session_id
                )
                
                result = await conn.execute(
                    'DELETE FROM chat_sessions WHERE session_id = $1',
                    session_id
                )
                
            return 'DELETE' in result and result.split()[-1] != '0'
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session.
        """
        async with self.pg_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT * FROM chat_sessions WHERE session_id = $1',
                session_id
            )
            
            if not row:
                return None
                
            info = dict(row)
            info['metadata'] = json.loads(info['metadata'])
            
            return info

    async def sync_session_message_counts(self) -> bool:
        """
        Sync message_count for all sessions (maintenance function).
        """
        async with self.pg_manager.pool.acquire() as conn:
            try:
                await conn.execute('''
                    UPDATE chat_sessions 
                    SET message_count = (
                        SELECT COUNT(*) 
                        FROM chat_messages 
                        WHERE chat_messages.session_id = chat_sessions.session_id
                    )
                ''')
                logger.info("Successfully synced message counts for all sessions")
                return True
            except Exception as e:
                logger.error(f"Error syncing message counts: {e}")
                return False