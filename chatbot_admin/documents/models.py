from django.db import models
from django.utils import timezone
from django.db.models import JSONField
from django.contrib.postgres.fields import ArrayField
import uuid

class RealArrayField(ArrayField):
    
    def db_type(self, connection):
        return 'real[]'

class Document(models.Model):
    
    document_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="ID tài liệu"
    )
    title = models.TextField(
        null=True,
        blank=True,
        verbose_name="Tiêu đề tài liệu"
    )
    
    file_name = models.TextField(
        verbose_name="Tên file"
    )
    
    file_type = models.TextField(
        verbose_name="Loại file"
    )
    
    file_size = models.IntegerField(
        verbose_name="Kích thước file"
    )
    
    user_id = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        to_field='user_id',
        db_column='user_id',
        verbose_name="Người dùng"
    )
    
    language = models.TextField(
        null=True,
        blank=True,
        verbose_name="Ngôn ngữ"
    )
    
    description = models.TextField(
        null=True,
        blank=True,
        verbose_name="Mô tả"
    )
    
    status = models.TextField(
        default='processing',
        verbose_name="Trạng thái"
    )
    
    chunks_count = models.IntegerField(
        default=0,
        verbose_name="Số lượng chunk"
    )
    
    chunk_size = models.IntegerField(
        verbose_name="Kích thước chunk"
    )
    
    chunk_overlap = models.IntegerField(
        verbose_name="Độ chồng lắp chunk"
    )
    
    metadata = JSONField(
        default=dict,
        verbose_name="Dữ liệu bổ sung"
    )
    
    file_path = models.TextField(
        null=True,
        blank=True,
        verbose_name="Đường dẫn file"
    )
    
    created_at = models.CharField(
        max_length=50,
        default=timezone.now,
        verbose_name="Thời gian tạo"
    )
    
    updated_at = models.CharField(
        max_length=50,
        default=timezone.now,
        verbose_name="Thời gian cập nhật"
    )
    
    class Meta:
        db_table = 'documents'
        verbose_name = "Tài liệu"
        verbose_name_plural = "Tài liệu"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user_id']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.title or self.file_name}"
    
    def save(self, *args, **kwargs):
        if not self.pk:
            self.created_at = timezone.now()
        self.updated_at = timezone.now()
        super().save(*args, **kwargs)


class DocumentChunk(models.Model):
    
    chunk_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="ID chunk"
    )
    
    document_id = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        to_field='document_id',
        db_column='document_id',
        verbose_name="Tài liệu"
    )
    
    content_path = models.TextField(
        verbose_name="Đường dẫn nội dung"
    )
    
    metadata = JSONField(
        default=dict,
        verbose_name="Dữ liệu bổ sung"
    )
    
    index = models.IntegerField(
        verbose_name="Chỉ mục"
    )
    
    chunk_size = models.IntegerField(
        verbose_name="Kích thước chunk"
    )
    
    prev_chunk_id = models.UUIDField(
        null=True,
        blank=True,
        verbose_name="ID chunk trước đó",
        help_text="ID của chunk trước đó để mở rộng context"
    )
    
    next_chunk_id = models.UUIDField(
        null=True,
        blank=True,
        verbose_name="ID chunk tiếp theo", 
        help_text="ID của chunk tiếp theo để mở rộng context"
    )
    
    created_at = models.CharField(
        max_length=50,
        default=timezone.now,
        verbose_name="Thời gian tạo"
    )
    
    class Meta:
        db_table = 'document_chunks'
        verbose_name = "Chunk tài liệu"
        verbose_name_plural = "Chunk tài liệu"
        ordering = ['document_id', 'index']
        indexes = [
            models.Index(fields=['document_id']),
            models.Index(fields=['document_id', 'index']),
            models.Index(fields=['prev_chunk_id']),
            models.Index(fields=['next_chunk_id']),
        ]
    
    def __str__(self):
        return f"Chunk {self.index} của {self.document_id.file_name}"

