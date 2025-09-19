import string
import secrets

from io import BytesIO

from django.shortcuts import render, redirect
from django.contrib.auth import (
    logout as auth_logout,
    authenticate,
    login as auth_login,
)
from django.http import Http404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.core.mail import send_mail
from django.contrib.auth.base_user import BaseUserManager

from documents.models import Document
from .services.document_api import DocumentAPI
from utils.s3 import logger
from .forms import DocumentForm
from utils.otp import generate_otp
from users.models import User


def login_view(request):
    if request.user.is_authenticated and request.user.is_superuser:
        return redirect('/admin/')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            if not user.is_active:
                messages.error(request, 'Tài khoản của bạn đã bị khóa!')
                return redirect('core:login')
            if not user.is_superuser:
                messages.error(request, 'Bạn không có quyền truy cập vào trang quản trị!')
                return redirect('core:login')

            auth_login(request, user)
            messages.success(request, 'Đăng nhập thành công!')
            return redirect('/admin/users/user')
        else:
            messages.error(request, 'Tên đăng nhập hoặc mật khẩu không đúng!')
    return render(request, 'admin/login.html')


@login_required(login_url='core:login')
def logout_view(request):
    auth_logout(request)
    response = redirect('core:login')
    response.delete_cookie('theme')
    messages.success(request, 'Đăng xuất thành công!')
    return response

def random_password(length=8):
    """
    Generate a random password of specified length.
    """
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def reset_password(request, user_id):
    try:
        new_password = random_password(8)
        user = User.objects.get(pk=user_id)
        user.set_password(new_password)
        send_mail(
            subject='Mật khẩu mới cho tài khoản của bạn',
            message=f'Mật khẩu mới của bạn là: {new_password}',
            from_email = settings.EMAIL_HOST_USER,
            recipient_list=[user.email],
            fail_silently=False,
        )
        user.save()
        messages.success(request, f'Mật khẩu đã được đặt lại thành công cho người dùng {user.username}!')
    except User.DoesNotExist:
        messages.error(request, 'Người dùng không tồn tại!')
    except Exception as e:
        messages.error(request, f'Đã xảy ra lỗi khi đặt lại mật khẩu: {str(e)}')
        logger.error('Error resetting password: %s', str(e))
    return redirect('core:admin')


@login_required(login_url='core:login')
def add_document(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():

            uploaded_file = request.FILES.getlist('file')

            if not uploaded_file:
                messages.error(request, 'Vui lòng chọn ít nhất một tệp để tải lên.')
                return render(request, 'admin/documents/change_form_document.html', {'form': form})

            try:
                files_for_api = []
                for file_obj in uploaded_file:
                    file_obj.seek(0)
                    files_for_api.append(('files', (file_obj.name, BytesIO(file_obj.read()), file_obj.content_type)))

                metadata = {
                    'title': form.cleaned_data.get('title', file_obj.name),
                    'description': form.cleaned_data.get('description', ''),
                    'language': form.cleaned_data.get('language', 'vi'),
                    'chunk_size': form.cleaned_data.get('chunk_size') or 1000,
                    'chunk_overlap': form.cleaned_data.get('chunk_overlap') or 100,
                }

                headers = {
                    'User-ID': str(request.user.user_id),
                    'otp': str(generate_otp()),
                }

                response = DocumentAPI.batch_upload_documents(headers, metadata, files_for_api)

                messages.success(request, f'Tài liệu đã được tải lên thành công!: {response.get("filename", "Tài liệu")}')
                return redirect('/admin/documents/document/')
            except Exception as e:
                messages.error(request, f'Đã xảy ra lỗi: {str(e)}')
        else:
            messages.error(request, 'Dữ liệu không hợp lệ. Vui lòng kiểm tra lại.')
    else:
        form = DocumentForm()
    return render(request, 'admin/documents/change_form_document.html', {'form': form})


@login_required(login_url='core:login')
def delete_document(request, document_id):
    if request.method == 'POST':
        try:
            headers = {
                'document_id': str(document_id),
                'User-ID': str(request.user.user_id),
                'otp': str(generate_otp()),
            }
            response = DocumentAPI.delete_document(headers, document_id)
            messages.success(request, f'Tài liệu đã được xóa thành công!: {response.get("message", "Tài liệu")}')
        except Exception as e:
            messages.error(request, f'Đã xảy ra lỗi: {str(e)}')
    return redirect('core:admin_document')


@login_required(login_url='core:login')
def download_document(request, document_id):
    try:
        try:
            Document.objects.get(document_id=document_id)
        except Document.DoesNotExist:
            messages.error(request, 'Tài liệu không tồn tại!')
            return redirect('/admin/documents/document/')

        download_url = DocumentAPI.download_document(document_id)

        if download_url:
            return redirect(download_url)
        else:
            logger.error(f"Không tạo được URL tải xuống cho tài liệu ID: {document_id}")
            messages.error(request, 'Không thể tạo URL tải xuống cho tài liệu này!')
            return redirect('core:admin_document')

    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        messages.error(request, 'Đã xảy ra lỗi khi tải file!')
        return redirect('/admin/documents/document/')


def password_change_done(request):
    messages.success(request, 'Mật khẩu đã được thay đổi thành công!')
    return redirect('/')


def admin_login(request):
    return redirect('/')


def admin_document_view(request):
    return redirect('/admin/documents/document/')


def admin_view(request):
    return redirect('/admin/users/user/')


def not_found(request, **kwargs):
    raise Http404("Page not found")