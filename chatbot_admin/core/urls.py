from django.urls import path

from . import views

app_name = 'core'

urlpatterns = [
    path("", views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('admin/', views.admin_view, name='admin'),
    path('admin/users/user/<str:user_id>/reset_password', views.reset_password, name='reset_password'),
    path('admin/password_change/done/', views.password_change_done, name='password_change_done'),

    # Document management URLs
    path('admin/documents/document/add/', views.add_document, name='add_document'),
    path('admin/documents/<str:document_id>/delete', views.delete_document, name='delete_document'),
    path('admin/documents/<str:document_id>/download', views.download_document, name='download_document'),

    # Admin URLs
    path('admin/login', views.admin_login, name='admin_login'),
    path('admin/users', views.admin_view, name='admin_users'),
    path('admin/documents/', views.admin_document_view, name='admin_document'),

    # Views for not found
    path("admin/auth/group/", views.not_found),
    path("admin/auth", views.not_found),
    path("admin/sites/site/", views.not_found),
    path("admin/sites", views.not_found),
    path("admin/users/user/<str:user_id>/password/", views.not_found),

]
