import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_admin.settings')
django.setup()

from django.conf import settings
from django.contrib.sites.models import Site
from utils.logger import get_logger
from django.contrib.auth import get_user_model

logger = get_logger(__name__)
User = get_user_model()

def create_superuser_and_site():
    username = os.getenv('ROOT_USER')
    email = os.getenv('ROOT_USER_EMAIL')
    password = os.getenv('ROOT_USER_PASSWORD')
    if not User.objects.filter(username=username).exists():
        User.objects.create_superuser(
            username=username,
            email=email,
            password=password
        )
        logger.info("Superuser created successfully")
    else:
        logger.info("Superuser already exists, skipping creation")

def create_site():
    site_domain = os.getenv('SITE_DOMAIN', 'localhost:8000')
    site_name = os.getenv('SITE_NAME', 'Development')
    site_id = getattr(settings, 'SITE_ID', 1)
    if not Site.objects.filter(id=site_id).exists():
        site, created = Site.objects.get_or_create(
            id=site_id,
            domain=site_domain,
            name=site_name
        )
        site.save()


if __name__ == '__main__':
    create_superuser_and_site()
    create_site()