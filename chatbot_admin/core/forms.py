from django import forms
from django.forms.widgets import FileInput

class MultipleFileInput(FileInput):
    """Custom widget để hỗ trợ multiple file uploads"""
    allow_multiple_selected = True
    
    def __init__(self, attrs=None):
        if attrs is None:
            attrs = {}
        attrs.update({'multiple': True})
        super().__init__(attrs)

class MultipleFileField(forms.FileField):
    """Custom field để xử lý multiple files"""
    widget = MultipleFileInput
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)
    
    def clean(self, data, initial=None):
        # Override clean method để xử lý multiple files
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result
        
class DocumentForm(forms.Form):

    file = MultipleFileField(
        label="Tải lên tài liệu",
        required=False,
        help_text="Chọn một hoặc nhiều tệp tài liệu để tải lên. Hỗ trợ định dạng: PDF, DOCX, TXT, XLSX, CSV.",
    )
    description = forms.CharField(
        label="Mô tả",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nhập mô tả cho tài liệu'}),
        required=False,
    )
    language = forms.ChoiceField(
        label="Ngôn ngữ",
        widget=forms.Select(attrs={
            'class': 'form-control',
        }),
        choices=[
            ('vi', 'Tiếng Việt'),
            ('en', 'Tiếng Anh'),
        ],
        required=False,
    )
    chunk_size = forms.IntegerField(
        label="Kích thước Chunk",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
        }),
        required=False,
    )
    chunk_overlap = forms.IntegerField(
        label="Độ chồng lắp Chunk",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
        }),
        required=False,
    )

    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data