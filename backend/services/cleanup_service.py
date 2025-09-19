import os
import asyncio
import time
from pathlib import Path
from typing import List, Tuple
from utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)

class CleanupService:
    
    def __init__(self):
        self.settings = get_settings()
        self.uploads_dir = self.settings.UPLOAD_DIR
        self.cleanup_interval = 3600
        self.max_file_age_hours = 24  
        self.running = False
        
    async def start_background_cleanup(self):
        if self.running:
            logger.warning("Cleanup service already running")
            return
            
        self.running = True
        logger.info("Starting background cleanup service")
        
        asyncio.create_task(self._cleanup_loop())
    
    async def stop_background_cleanup(self):
        self.running = False
        logger.info("Stopping background cleanup service")
    
    async def _cleanup_loop(self):
        while self.running:
            try:
                await self.cleanup_old_uploads()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def cleanup_old_uploads(self) -> Tuple[int, int]:
        if not os.path.exists(self.uploads_dir):
            logger.debug(f"Uploads directory does not exist: {self.uploads_dir}")
            return 0, 0
        
        files_removed = 0
        bytes_freed = 0
        cutoff_time = time.time() - (self.max_file_age_hours * 3600)
        
        try:
            for file_path in Path(self.uploads_dir).rglob("*"):
                if not file_path.is_file():
                    continue
                
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_size = file_path.stat().st_size
                        os.remove(file_path)
                        
                        files_removed += 1
                        bytes_freed += file_size
                        
                        file_age_hours = (time.time() - file_path.stat().st_mtime) / 3600
                        logger.debug(f"Removed old upload file: {file_path.name} "
                                   f"(age: {file_age_hours:.1f}h, size: {file_size} bytes)")
                        
                    except Exception as e:
                        logger.warning(f"Failed to remove file {file_path}: {e}")
            
            if files_removed > 0:
                mb_freed = bytes_freed / (1024 * 1024)
                logger.info(f"Cleanup completed: removed {files_removed} files, "
                           f"freed {mb_freed:.2f} MB")
            
            return files_removed, bytes_freed
            
        except Exception as e:
            logger.error(f"Error during uploads cleanup: {e}")
            return 0, 0
    
    async def cleanup_specific_file(self, file_path: str) -> bool:
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                logger.debug(f"Removed specific file: {file_path} ({file_size} bytes)")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to remove specific file {file_path}: {e}")
            return False
    
    async def get_uploads_stats(self) -> dict:
        if not os.path.exists(self.uploads_dir):
            return {
                "total_files": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "old_files": 0,
                "old_files_size_bytes": 0,
                "old_files_size_mb": 0
            }
        
        total_files = 0
        total_size = 0
        old_files = 0
        old_files_size = 0
        cutoff_time = time.time() - (self.max_file_age_hours * 3600)
        
        try:
            for file_path in Path(self.uploads_dir).rglob("*"):
                if not file_path.is_file():
                    continue
                
                file_size = file_path.stat().st_size
                total_files += 1
                total_size += file_size
                
                if file_path.stat().st_mtime < cutoff_time:
                    old_files += 1
                    old_files_size += file_size
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "old_files": old_files,
                "old_files_size_bytes": old_files_size,
                "old_files_size_mb": round(old_files_size / (1024 * 1024), 2),
                "cleanup_threshold_hours": self.max_file_age_hours
            }
            
        except Exception as e:
            logger.error(f"Error getting uploads stats: {e}")
            return {
                "error": str(e)
            }

cleanup_service = CleanupService() 