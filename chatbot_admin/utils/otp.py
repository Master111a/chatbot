import pyotp
from django.conf import settings


OTP_SECRET_KEY = settings.OTP_SECRET_KEY
OTP_VALIDITY_SECONDS = int(settings.OTP_VALIDITY_SECONDS)

def generate_otp() -> str:
    """
    Generate a one-time password (OTP) using the provided secret.
    """
    if not OTP_SECRET_KEY:
        raise ValueError("OTP_SECRET_KEY is not set in settings.")

    totp = pyotp.TOTP(OTP_SECRET_KEY,8,  interval=OTP_VALIDITY_SECONDS)
    return totp.now()