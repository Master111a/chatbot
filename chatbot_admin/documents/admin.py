from django.contrib import admin
from django.utils.safestring import mark_safe
from django.urls import reverse, NoReverseMatch
from django.middleware.csrf import get_token
from .models import Document


class DocumentAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'file_size',
                    'user_id', 'language', 'status', 'action_buttons')
    search_fields = ('file_name', 'file_type', 'language', 'description')
    list_filter = ('language', 'status', 'user_id')
    list_per_page = 10
    fieldsets = (
        ('Thông tin tài liệu', {
            'fields': ('file_name', 'file_type', 'file_size', 'language', 'description',
                        'status', 'chunk_size', 'chunks_count', 'chunk_overlap'),
        }),
        ('Thông tin tạo và cập nhật', {
            'fields': ('user_id', 'created_at', 'updated_at'),
        })
    )

    def action_buttons(self, obj):
        try:
            delete_url = reverse('core:delete_document', kwargs={'document_id': obj.document_id})
            download_url = reverse('core:download_document', kwargs={'document_id': obj.document_id})
        except NoReverseMatch:
            delete_url = f'/admin/documents/{obj.document_id}/delete/'
            download_url = f'/admin/documents/{obj.document_id}/download/'

        request = getattr(self, '_current_request', None)
        csrf_token = get_token(request) if request else ''

        download_button = f'''
            <a href="{download_url}"
                class="btn btn-success btn-sm"
                title="Tải xuống"
                style="margin-right: 5px;">
                <i class="fas fa-download"></i> Tải xuống
            </a>
        '''
        delete_button = f'''
            <button type="button" class="btn btn-danger btn-sm"
                onclick="deleteDocument('{delete_url}', '{csrf_token}', '{obj.file_name}')">
                <i class="fas fa-trash"></i> Xóa
            </button>
        '''

        actions_html = download_button + delete_button
        return mark_safe(actions_html)

    action_buttons.short_description = 'Hành động'

    def has_add_permission(self, request):
        return True

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def changelist_view(self, request, extra_context=None):
        self._current_request = request
        if not extra_context:
            extra_context = {}
        extra_context['title'] = 'Danh sách tài liệu'
        return super().changelist_view(request, extra_context=extra_context)

    change_list_template = 'admin/documents/document_list.html'


admin.site.register(Document, DocumentAdmin)
