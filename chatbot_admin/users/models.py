from django.db import models
from django.contrib.auth.models import AbstractUser
# from django.utils import timezone
import uuid


class User(AbstractUser):
    """Model người dùng hệ thống chatbot"""

    user_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="ID người dùng"
    )

    email = models.EmailField(
        unique=True,
        verbose_name="Email"
    )

    username = models.CharField(
        max_length=100,
        unique=True,
        verbose_name="Tên đăng nhập"
    )

    name = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        verbose_name="Họ tên"
    )

    avatar_url = models.TextField(
        null=True,
        blank=True,
        verbose_name="URL ảnh đại diện"
    )

    gitlab_id = models.IntegerField(
        unique=True,
        null=True,
        blank=True,
        verbose_name="GitLab ID"
    )

    password = models.TextField(
        verbose_name="Mật khẩu",
        null=True,
        blank=True
    )

    is_staff = models.BooleanField(
        default=False,
        verbose_name="Quyền nhân viên",
        help_text="Chỉ định người dùng nào được phép truy cập vào trang admin."
    )

    is_active = models.BooleanField(
        default=True,
        verbose_name="Kích hoạt tài khoản",
        help_text="Chỉ định người dùng nào có thể đăng nhập vào hệ thống."
    )

    is_superuser = models.BooleanField(
        default=False,
        verbose_name="Quyền quản trị viên",
        help_text="Chỉ định người dùng nào có quyền quản trị cao nhất."
    )
    first_name = None
    last_name = None

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']

    class Meta:
        db_table = 'users'
        verbose_name = "Người dùng"
        verbose_name_plural = "Người dùng"
        ordering = ['-date_joined']

    def __str__(self):
        return f"{self.name or self.username} ({self.email})"
