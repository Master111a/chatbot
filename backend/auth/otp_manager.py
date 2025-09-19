import base64
import hmac
import hashlib
import time
import struct
from typing import Optional
from fastapi import HTTPException, Header
from config.settings import get_settings
from utils.logger import get_logger
    
logger = get_logger(__name__)

settings = get_settings()

class OTPManager:
    """
    Manager for admin OTP generation and validation using TOTP algorithm.
    Compatible with pyotp library for 8-digit OTP codes.
    """
    
    def __init__(self):
        """
        Initialize OTP Manager with secret key from settings and 30 second validity window.
        """
        self.secret_key = settings.OTP_SECRET_KEY
        self.validity_seconds = 30
        self.digits = 8
        self.tolerance_windows = 2
    
    def _get_hotp_token(self, counter: int) -> str:
        """
        Generate HOTP token based on counter and secret key.
        Compatible with pyotp implementation.
        
        Args:
            counter: Time-based counter value
            
        Returns:
            8-digit OTP token
        """
        try:
            secret = self.secret_key.upper()
            missing_padding = len(secret) % 8
            if missing_padding:
                secret += '=' * (8 - missing_padding)
            
            key = base64.b32decode(secret, True)
            msg = struct.pack(">Q", counter)
            
            h = hmac.new(key, msg, hashlib.sha1).digest()
            o = h[19] & 15
            
            h = (struct.unpack(">I", h[o:o+4])[0] & 0x7fffffff) % 10**self.digits
            return str(h).zfill(self.digits)
        except Exception as e:
            logger.error(f"Error generating HOTP token: {e}")
            raise ValueError(f"Failed to generate OTP token: {e}")
    
    def get_current_totp_token(self) -> str:
        """
        Generate current time-based OTP token for testing purposes.
        
        Returns:
            Current 8-digit OTP token
        """
        counter = int(time.time() // self.validity_seconds)
        return self._get_hotp_token(counter)
    
    def verify_totp(self, token: str, timestamp: Optional[int] = None) -> bool:
        """
        Verify if the 8-digit token is valid for current time window with tolerance.
        Accepts tokens from previous and next time windows for clock drift tolerance.
        Compatible with pyotp generated tokens.
        
        Args:
            token: 8-digit OTP token from pyotp or Django admin
            timestamp: Optional specific timestamp to verify against
            
        Returns:
            True if token is valid within tolerance window
        """
        if not token or len(token) != self.digits or not token.isdigit():
            logger.warning(f"Invalid OTP format: {token}")
            return False
            
        if timestamp is None:
            timestamp = int(time.time())
        
        counter = timestamp // self.validity_seconds
        
        for i in range(-self.tolerance_windows, self.tolerance_windows + 1):
            try:
                test_counter = counter + i
                expected_token = self._get_hotp_token(test_counter)
                if expected_token == token:
                    logger.info(f"Valid OTP verified: {token} (window offset: {i})")
                    return True
            except Exception as e:
                logger.error(f"Error verifying OTP for counter {test_counter}: {e}")
                continue
                
        logger.warning(f"Invalid OTP: {token} at timestamp {timestamp}")
        return False
    
    def get_time_remaining(self) -> int:
        """
        Get remaining seconds until current OTP expires.
        
        Returns:
            Seconds remaining in current time window
        """
        return self.validity_seconds - (int(time.time()) % self.validity_seconds)

otp_manager = OTPManager()


async def verify_admin_headers(
    otp: str = Header(..., description="OTP code in header"),
    user_id: str = Header(..., description="User ID in header", alias="User-ID")
) -> tuple[str, str]:
    """
    Dependency function to verify OTP and validate user has admin privileges.
    
    Args:
        otp: OTP code from header
        user_id: User ID from header (with alias User-ID)
        
    Returns:
        Tuple of (otp, user_id) if verification is successful
        
    Raises:
        HTTPException: If OTP is invalid, expired, or user is not admin
    """
    if not otp_manager.verify_totp(otp):
        raise HTTPException(status_code=401, detail="Invalid or expired OTP")
    
    try:
        from db.pg_manager import UserManager
        
        user_manager = UserManager()
        user = await user_manager.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=404, 
                detail=f"User not found with ID: {user_id}"
            )
        
        if not user.get("is_superuser", False):
            raise HTTPException(
                status_code=403, 
                detail="Admin privileges required (is_superuser=True)"
            )
        
        logger.info(f"Valid OTP and admin privileges verified for user {user_id}")
        return otp, user_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying user admin status: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Error verifying user privileges"
        )