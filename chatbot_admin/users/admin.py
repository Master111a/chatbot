from django.contrib import admin
from django.conf import settings

from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import Permission
from .models import User


class UserAdmin(BaseUserAdmin):
    list_display = ('username', 'email', 'name', 'is_staff', 'is_active', 'is_superuser', 'last_login', 'date_joined')
    search_fields = ('username', 'email')
    ordering = ('-date_joined',)
    list_filter = ()
    add_fieldsets = (
        ('Thêm người dùng mới', {
            'fields': ('username', 'name', 'email', 'password1', 'password2'),
            'classes': ('wide',)
        }),
    )

    fieldsets = (
        ('Thông tin người dùng', {
            'fields': ('username', 'name', 'email')
        }),
        ('Trạng thái tài khoản', {
            'fields': ('is_staff', 'is_active', 'is_superuser')
        }),
    )

    def save_model(self, request, obj, form, change):
        if request.user.username == settings.ROOT_USER:
            if obj.is_superuser:
                obj.is_staff = True

        if not change:
            obj.is_staff = True
            super().save_model(request, obj, form, change)
            all_perms = Permission.objects.exclude(
                content_type__app_label='auth')
            obj.user_permissions.set(all_perms)
        else:
            super().save_model(request, obj, form, change)

    def get_fieldsets(self, request, obj=None):
        if not obj:
            return self.add_fieldsets

        if request.user.username == obj.username:
            return (
                (None, {'fields': ('username',)}),
                ("Thông tin người dùng", {'fields': ('name', 'email')}),
                ("Trạng thái tài khoản", {'fields': ('is_staff', 'is_active', 'is_superuser')}),
            )
        if request.user.username == settings.ROOT_USER:
            return (
                (None, {'fields': ('username',)}),
                ("Thông tin người dùng", {'fields': ('name', 'email')}),
                ("Trạng thái tài khoản", {'fields': ('is_staff', 'is_active', 'is_superuser')}),
            )
        return (
            (None, {'fields': ('username',)}),
            ("Thông tin người dùng", {'fields': ('name', 'email')}),
            ("Trạng thái tài khoản", {'fields': ('is_staff', 'is_active', 'is_superuser')})
        )

    def has_add_permission(self, request):
        return request.user.is_superuser

    def has_delete_permission(self, request, obj=None):
        if obj is None:
            return request.user.is_superuser

        if request.user.username == settings.ROOT_USER:
            if obj.username == settings.ROOT_USER:
                return False
            return True

        if request.user.is_superuser and obj.username != settings.ROOT_USER and request.user != obj:
            if obj.is_superuser:
                return False
            return True

        return False

    def has_view_permission(self, request, obj=None):
        return (request.user.is_active
            and (request.user.is_staff
                or request.user.is_superuser))

    def has_change_permission(self, request, obj=None):
        if obj is None:
            return request.user.is_superuser

        if request.user.username == settings.ROOT_USER:
            return True

        if obj.is_superuser and request.user.is_superuser and request.user != obj:
            return False

        return request.user.is_superuser

    def get_readonly_fields(self, request, obj=None):
        if obj and request.user == obj:
            return ('is_active', 'is_staff', 'is_superuser', 'last_login', 'date_joined')

        if request.user.username == settings.ROOT_USER and obj and obj.username != settings.ROOT_USER:
            return ('user_permissions', 'last_login', 'date_joined')

        if request.user.username == settings.ROOT_USER and obj and obj.username == settings.ROOT_USER:
            return ('user_permissions', 'last_login', 'date_joined')

        return ('user_permissions', 'last_login', 'date_joined')

admin.site.register(User, UserAdmin)
