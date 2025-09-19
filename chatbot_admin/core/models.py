from django.db import models
from django.utils import timezone
from django.db.models import JSONField
import uuid


class ChatSession(models.Model):
    """Model phiên trò chuyện"""
    
    session_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="ID phiên"
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
    
    title = models.TextField(
        null=True,
        blank=True,
        verbose_name="Tiêu đề"
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
    
    message_count = models.IntegerField(
        default=0,
        verbose_name="Số tin nhắn"
    )
    
    metadata = JSONField(
        default=dict,
        verbose_name="Dữ liệu bổ sung"
    )
    
    is_anonymous = models.BooleanField(
        default=True,
        verbose_name="Ẩn danh"
    )
    
    class Meta:
        db_table = 'chat_sessions'
        verbose_name = "Phiên trò chuyện"
        verbose_name_plural = "Phiên trò chuyện"
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.title or 'Phiên trò chuyện'} - {self.session_id[:8]}"
    
    def save(self, *args, **kwargs):
        if not self.pk:
            self.created_at = timezone.now()
        self.updated_at = timezone.now()
        super().save(*args, **kwargs)


class ChatMessage(models.Model):
    """Model tin nhắn trò chuyện"""
    
    message_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="ID tin nhắn"
    )
    
    session_id = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        to_field='session_id',
        db_column='session_id',
        verbose_name="Phiên trò chuyện"
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
    
    message = models.TextField(
        verbose_name="Tin nhắn"
    )
    
    response = models.TextField(
        verbose_name="Phản hồi"
    )
    
    timestamp = models.CharField(
        max_length=50,
        default=timezone.now,
        verbose_name="Thời gian"
    )
    
    metadata = JSONField(
        default=dict,
        verbose_name="Dữ liệu bổ sung"
    )
    
    class Meta:
        db_table = 'chat_messages'
        verbose_name = "Tin nhắn trò chuyện"
        verbose_name_plural = "Tin nhắn trò chuyện"
        ordering = ['timestamp']
        indexes = [
            models.Index(fields=['session_id']),
            models.Index(fields=['user_id']),
            models.Index(fields=['timestamp']),
        ]
    
    def __str__(self):
        return f"Tin nhắn {self.message_id[:8]} - {self.message[:50]}..."
