"""
Memory Backup and Recovery System

This module provides comprehensive backup and recovery mechanisms for the memory system,
including automated backups, incremental backups, compression, encryption, and full
system restoration capabilities.
"""

import json
import logging
import shutil
import tarfile
import gzip
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import os

# Handle imports gracefully for both package and standalone execution
try:
    from .file_operations import MemoryFileOperations, read_memory_file, write_memory_file
    from .memory_analytics import MemoryAnalytics
except ImportError:
    # Fallback for standalone execution
    from file_operations import MemoryFileOperations, read_memory_file, write_memory_file
    from memory_analytics import MemoryAnalytics

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups supported by the system."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Status of backup operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CompressionType(Enum):
    """Compression types for backups."""
    NONE = "none"
    GZIP = "gzip"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"


@dataclass
class BackupConfig:
    """Configuration for backup operations."""
    backup_dir: str
    max_backups: int = 10
    compression: CompressionType = CompressionType.TAR_GZ
    encrypt_backups: bool = False
    encryption_key: Optional[str] = None
    auto_backup_interval_hours: int = 24
    incremental_backup_interval_hours: int = 6
    verify_backups: bool = True
    backup_analytics: bool = True
    backup_system_files: bool = True
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = ['*.tmp', '*.log', '__pycache__', '.DS_Store']


@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    backup_id: str
    backup_type: BackupType
    created_at: str
    completed_at: Optional[str]
    status: BackupStatus
    file_count: int
    total_size_bytes: int
    compressed_size_bytes: int
    checksum: str
    base_backup_id: Optional[str]  # For incremental backups
    files_included: List[str]
    compression_ratio: float
    backup_path: str
    verification_status: bool
    error_message: Optional[str] = None


@dataclass
class RestoreOptions:
    """Options for restore operations."""
    target_directory: str
    backup_id: str
    restore_type: str = "full"  # full, selective, point_in_time
    selected_files: Optional[List[str]] = None
    overwrite_existing: bool = False
    verify_integrity: bool = True
    restore_permissions: bool = True
    create_restore_log: bool = True


@dataclass
class RestoreResult:
    """Result of a restore operation."""
    restore_id: str
    backup_id: str
    started_at: str
    completed_at: str
    status: str
    files_restored: int
    files_failed: int
    total_size_restored: int
    verification_passed: bool
    error_messages: List[str]
    restore_log_path: Optional[str]


class MemoryBackupSystem:
    """
    Comprehensive memory backup and recovery system.
    
    This class provides automated backup creation, incremental backups,
    compression, encryption, and full system restoration capabilities
    to protect against data loss and enable disaster recovery.
    """
    
    def __init__(self, memory_base_path: str, config: Optional[BackupConfig] = None):
        """
        Initialize the backup system.
        
        Args:
            memory_base_path: Base path for memory storage
            config: Backup configuration (uses defaults if None)
        """
        self.memory_base_path = Path(memory_base_path)
        self.config = config or BackupConfig(
            backup_dir=str(self.memory_base_path / 'system' / 'backups')
        )
        
        # Setup backup directory
        self.backup_dir = Path(self.config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_dir = self.backup_dir / 'metadata'
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Restore logs
        self.restore_logs_dir = self.backup_dir / 'restore_logs'
        self.restore_logs_dir.mkdir(exist_ok=True)
        
        # Initialize backup database
        self.db_path = self.backup_dir / 'backup_registry.db'
        self.init_database()
        
        # Auto-backup thread
        self._auto_backup_thread = None
        self._auto_backup_running = False
        self._backup_lock = threading.Lock()
        
        # File operations
        self.file_ops = MemoryFileOperations()
        
    def init_database(self) -> None:
        """Initialize the backup registry database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Backups table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS backups (
                        backup_id TEXT PRIMARY KEY,
                        backup_type TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        completed_at TEXT,
                        status TEXT NOT NULL,
                        file_count INTEGER,
                        total_size_bytes INTEGER,
                        compressed_size_bytes INTEGER,
                        checksum TEXT,
                        base_backup_id TEXT,
                        backup_path TEXT,
                        compression_ratio REAL,
                        verification_status BOOLEAN,
                        error_message TEXT,
                        metadata_json TEXT
                    )
                ''')
                
                # Restore operations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS restore_operations (
                        restore_id TEXT PRIMARY KEY,
                        backup_id TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        status TEXT NOT NULL,
                        files_restored INTEGER,
                        files_failed INTEGER,
                        total_size_restored INTEGER,
                        verification_passed BOOLEAN,
                        error_messages TEXT,
                        restore_log_path TEXT,
                        options_json TEXT
                    )
                ''')
                
                # File tracking table for incremental backups
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS file_tracking (
                        file_path TEXT PRIMARY KEY,
                        last_modified TEXT NOT NULL,
                        last_backup_id TEXT,
                        file_size INTEGER,
                        checksum TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_backups_created ON backups(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_backups_type ON backups(backup_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_restore_started ON restore_operations(started_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_tracking_modified ON file_tracking(last_modified)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing backup database: {e}")
    
    def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        description: Optional[str] = None
    ) -> BackupMetadata:
        """
        Create a backup of the memory system.
        
        Args:
            backup_type: Type of backup to create
            description: Optional description for the backup
            
        Returns:
            Backup metadata
        """
        with self._backup_lock:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting {backup_type.value} backup: {backup_id}")
            
            # Initialize metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                created_at=datetime.now().isoformat(),
                completed_at=None,
                status=BackupStatus.IN_PROGRESS,
                file_count=0,
                total_size_bytes=0,
                compressed_size_bytes=0,
                checksum="",
                base_backup_id=None,
                files_included=[],
                compression_ratio=0.0,
                backup_path="",
                verification_status=False
            )
            
            try:
                # Record backup start
                self._record_backup_metadata(metadata)
                
                # Determine files to backup
                if backup_type == BackupType.FULL:
                    files_to_backup = self._get_all_memory_files()
                elif backup_type == BackupType.INCREMENTAL:
                    base_backup = self._get_latest_backup(BackupType.FULL)
                    if not base_backup:
                        logger.warning("No full backup found, creating full backup instead")
                        backup_type = BackupType.FULL
                        files_to_backup = self._get_all_memory_files()
                    else:
                        metadata.base_backup_id = base_backup.backup_id
                        files_to_backup = self._get_changed_files_since_backup(base_backup.backup_id)
                elif backup_type == BackupType.DIFFERENTIAL:
                    base_backup = self._get_latest_backup(BackupType.FULL)
                    if not base_backup:
                        logger.warning("No full backup found, creating full backup instead")
                        backup_type = BackupType.FULL
                        files_to_backup = self._get_all_memory_files()
                    else:
                        metadata.base_backup_id = base_backup.backup_id
                        files_to_backup = self._get_changed_files_since_backup(base_backup.backup_id)
                else:  # SNAPSHOT
                    files_to_backup = self._get_all_memory_files()
                
                # Create backup archive
                backup_path = self.backup_dir / f"{backup_id}.{self.config.compression.value}"
                metadata.backup_path = str(backup_path)
                
                total_size = self._create_backup_archive(files_to_backup, backup_path)
                
                # Update metadata
                metadata.file_count = len(files_to_backup)
                metadata.total_size_bytes = total_size
                metadata.compressed_size_bytes = backup_path.stat().st_size if backup_path.exists() else 0
                metadata.compression_ratio = (
                    1.0 - (metadata.compressed_size_bytes / metadata.total_size_bytes)
                    if metadata.total_size_bytes > 0 else 0.0
                )
                metadata.files_included = [str(f) for f in files_to_backup]
                metadata.checksum = self._calculate_file_checksum(backup_path)
                metadata.completed_at = datetime.now().isoformat()
                metadata.status = BackupStatus.COMPLETED
                
                # Verify backup if configured
                if self.config.verify_backups:
                    metadata.verification_status = self._verify_backup(backup_path, files_to_backup)
                else:
                    metadata.verification_status = True
                
                # Update file tracking for incremental backups
                self._update_file_tracking(files_to_backup, backup_id)
                
                # Save metadata
                self._record_backup_metadata(metadata)
                self._save_backup_metadata_file(metadata)
                
                # Cleanup old backups
                self._cleanup_old_backups()
                
                logger.info(f"Backup {backup_id} completed successfully")
                return metadata
                
            except Exception as e:
                logger.error(f"Backup {backup_id} failed: {e}")
                metadata.status = BackupStatus.FAILED
                metadata.error_message = str(e)
                metadata.completed_at = datetime.now().isoformat()
                self._record_backup_metadata(metadata)
                raise
    
    def restore_backup(
        self,
        backup_id: str,
        options: Optional[RestoreOptions] = None
    ) -> RestoreResult:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of the backup to restore
            options: Restore options
            
        Returns:
            Restore result
        """
        restore_id = f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting restore operation: {restore_id} from backup: {backup_id}")
        
        # Default options
        if options is None:
            options = RestoreOptions(
                target_directory=str(self.memory_base_path),
                backup_id=backup_id
            )
        
        # Initialize result
        result = RestoreResult(
            restore_id=restore_id,
            backup_id=backup_id,
            started_at=datetime.now().isoformat(),
            completed_at="",
            status="in_progress",
            files_restored=0,
            files_failed=0,
            total_size_restored=0,
            verification_passed=False,
            error_messages=[],
            restore_log_path=None
        )
        
        try:
            # Get backup metadata
            backup_metadata = self._get_backup_metadata(backup_id)
            if not backup_metadata:
                raise ValueError(f"Backup {backup_id} not found")
            
            # Create restore log
            if options.create_restore_log:
                log_path = self.restore_logs_dir / f"{restore_id}.log"
                result.restore_log_path = str(log_path)
            
            # Verify backup integrity
            if options.verify_integrity:
                if not self._verify_backup_integrity(backup_metadata):
                    raise ValueError(f"Backup {backup_id} failed integrity check")
            
            # Extract backup
            backup_path = Path(backup_metadata.backup_path)
            temp_extract_dir = self.backup_dir / f"temp_restore_{restore_id}"
            
            try:
                # Extract files
                extracted_files = self._extract_backup_archive(backup_path, temp_extract_dir)
                
                # Filter files if selective restore
                if options.restore_type == "selective" and options.selected_files:
                    extracted_files = [f for f in extracted_files if str(f) in options.selected_files]
                
                # Restore files
                target_dir = Path(options.target_directory)
                target_dir.mkdir(parents=True, exist_ok=True)
                
                for source_file in extracted_files:
                    try:
                        # Calculate target path
                        relative_path = source_file.relative_to(temp_extract_dir)
                        target_file = target_dir / relative_path
                        
                        # Check if file exists and handle overwrite
                        if target_file.exists() and not options.overwrite_existing:
                            logger.warning(f"Skipping existing file: {target_file}")
                            continue
                        
                        # Create target directory
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        shutil.copy2(source_file, target_file)
                        
                        # Restore permissions if requested
                        if options.restore_permissions:
                            shutil.copystat(source_file, target_file)
                        
                        result.files_restored += 1
                        result.total_size_restored += source_file.stat().st_size
                        
                        # Log to restore log
                        if result.restore_log_path:
                            with open(result.restore_log_path, 'a') as log_file:
                                log_file.write(f"RESTORED: {source_file} -> {target_file}\n")
                        
                    except Exception as e:
                        error_msg = f"Failed to restore {source_file}: {e}"
                        logger.error(error_msg)
                        result.error_messages.append(error_msg)
                        result.files_failed += 1
                        
                        if result.restore_log_path:
                            with open(result.restore_log_path, 'a') as log_file:
                                log_file.write(f"FAILED: {source_file} - {error_msg}\n")
                
                # Verify restoration if requested
                if options.verify_integrity:
                    result.verification_passed = self._verify_restored_files(
                        extracted_files, target_dir, temp_extract_dir
                    )
                else:
                    result.verification_passed = True
                
                result.status = "completed"
                result.completed_at = datetime.now().isoformat()
                
                logger.info(f"Restore {restore_id} completed: {result.files_restored} files restored")
                
            finally:
                # Cleanup temporary extraction directory
                if temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
            
            # Record restore operation
            self._record_restore_operation(result, options)
            
            return result
            
        except Exception as e:
            logger.error(f"Restore {restore_id} failed: {e}")
            result.status = "failed"
            result.error_messages.append(str(e))
            result.completed_at = datetime.now().isoformat()
            self._record_restore_operation(result, options)
            raise
    
    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        limit: int = 50
    ) -> List[BackupMetadata]:
        """
        List available backups.
        
        Args:
            backup_type: Filter by backup type
            limit: Maximum number of backups to return
            
        Returns:
            List of backup metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if backup_type:
                    cursor.execute('''
                        SELECT * FROM backups 
                        WHERE backup_type = ? 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (backup_type.value, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM backups 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (limit,))
                
                backups = []
                for row in cursor.fetchall():
                    backup = BackupMetadata(
                        backup_id=row[0],
                        backup_type=BackupType(row[1]),
                        created_at=row[2],
                        completed_at=row[3],
                        status=BackupStatus(row[4]),
                        file_count=row[5] or 0,
                        total_size_bytes=row[6] or 0,
                        compressed_size_bytes=row[7] or 0,
                        checksum=row[8] or "",
                        base_backup_id=row[9],
                        backup_path=row[10] or "",
                        compression_ratio=row[11] or 0.0,
                        verification_status=bool(row[12]) if row[12] is not None else False,
                        error_message=row[13],
                        files_included=[]  # Not stored in DB for performance
                    )
                    backups.append(backup)
                
                return backups
                
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: ID of the backup to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get backup metadata
            backup_metadata = self._get_backup_metadata(backup_id)
            if not backup_metadata:
                logger.warning(f"Backup {backup_id} not found")
                return False
            
            # Delete backup file
            backup_path = Path(backup_metadata.backup_path)
            if backup_path.exists():
                backup_path.unlink()
            
            # Delete metadata file
            metadata_file = self.metadata_dir / f"{backup_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM backups WHERE backup_id = ?', (backup_id,))
                cursor.execute('DELETE FROM file_tracking WHERE last_backup_id = ?', (backup_id,))
                conn.commit()
            
            logger.info(f"Backup {backup_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting backup {backup_id}: {e}")
            return False
    
    def start_auto_backup(self) -> None:
        """Start automatic backup scheduling."""
        if self._auto_backup_running:
            logger.warning("Auto-backup is already running")
            return
        
        self._auto_backup_running = True
        self._auto_backup_thread = threading.Thread(target=self._auto_backup_worker, daemon=True)
        self._auto_backup_thread.start()
        
        logger.info("Auto-backup started")
    
    def stop_auto_backup(self) -> None:
        """Stop automatic backup scheduling."""
        self._auto_backup_running = False
        if self._auto_backup_thread:
            self._auto_backup_thread.join(timeout=5)
        
        logger.info("Auto-backup stopped")
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """
        Get backup system statistics.
        
        Returns:
            Dictionary with backup statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total backups
                cursor.execute('SELECT COUNT(*) FROM backups')
                total_backups = cursor.fetchone()[0]
                
                # Backups by type
                cursor.execute('SELECT backup_type, COUNT(*) FROM backups GROUP BY backup_type')
                backups_by_type = dict(cursor.fetchall())
                
                # Total backup size
                cursor.execute('SELECT SUM(compressed_size_bytes) FROM backups WHERE status = "completed"')
                total_backup_size = cursor.fetchone()[0] or 0
                
                # Average compression ratio
                cursor.execute('SELECT AVG(compression_ratio) FROM backups WHERE status = "completed"')
                avg_compression_ratio = cursor.fetchone()[0] or 0.0
                
                # Recent backup activity
                cursor.execute('''
                    SELECT COUNT(*) FROM backups 
                    WHERE created_at >= datetime('now', '-7 days')
                ''')
                recent_backups = cursor.fetchone()[0]
                
                # Failed backups
                cursor.execute('SELECT COUNT(*) FROM backups WHERE status = "failed"')
                failed_backups = cursor.fetchone()[0]
                
                # Restore operations
                cursor.execute('SELECT COUNT(*) FROM restore_operations')
                total_restores = cursor.fetchone()[0]
                
                return {
                    'total_backups': total_backups,
                    'backups_by_type': backups_by_type,
                    'total_backup_size_bytes': total_backup_size,
                    'average_compression_ratio': avg_compression_ratio,
                    'recent_backups_7_days': recent_backups,
                    'failed_backups': failed_backups,
                    'total_restore_operations': total_restores,
                    'auto_backup_running': self._auto_backup_running,
                    'backup_directory': str(self.backup_dir),
                    'config': asdict(self.config)
                }
                
        except Exception as e:
            logger.error(f"Error getting backup statistics: {e}")
            return {}
    
    # Helper methods
    
    def _get_all_memory_files(self) -> List[Path]:
        """Get list of all memory files to backup."""
        files = []
        
        # Search in common memory directories
        search_dirs = ['interactions', 'core', 'archive', 'system']
        
        for dir_name in search_dirs:
            dir_path = self.memory_base_path / dir_name
            if dir_path.exists():
                for pattern in ['*.md', '*.json', '*.yaml', '*.yml']:
                    files.extend(dir_path.rglob(pattern))
        
        # Filter out excluded patterns
        filtered_files = []
        for file_path in files:
            if not any(file_path.match(pattern) for pattern in self.config.exclude_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _get_changed_files_since_backup(self, base_backup_id: str) -> List[Path]:
        """Get files that have changed since the specified backup."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT file_path, last_modified FROM file_tracking 
                    WHERE last_backup_id = ?
                ''', (base_backup_id,))
                
                tracked_files = {row[0]: row[1] for row in cursor.fetchall()}
            
            changed_files = []
            all_files = self._get_all_memory_files()
            
            for file_path in all_files:
                file_path_str = str(file_path)
                current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                
                if (file_path_str not in tracked_files or 
                    tracked_files[file_path_str] != current_mtime):
                    changed_files.append(file_path)
            
            return changed_files
            
        except Exception as e:
            logger.error(f"Error getting changed files: {e}")
            return self._get_all_memory_files()  # Fallback to full backup
    
    def _create_backup_archive(self, files: List[Path], archive_path: Path) -> int:
        """Create compressed backup archive."""
        total_size = 0
        
        if self.config.compression == CompressionType.TAR_GZ:
            with tarfile.open(archive_path, 'w:gz') as tar:
                for file_path in files:
                    if file_path.exists():
                        # Calculate relative path from memory base
                        try:
                            arcname = file_path.relative_to(self.memory_base_path)
                        except ValueError:
                            arcname = file_path.name
                        
                        tar.add(file_path, arcname=str(arcname))
                        total_size += file_path.stat().st_size
        
        elif self.config.compression == CompressionType.TAR_BZ2:
            with tarfile.open(archive_path, 'w:bz2') as tar:
                for file_path in files:
                    if file_path.exists():
                        try:
                            arcname = file_path.relative_to(self.memory_base_path)
                        except ValueError:
                            arcname = file_path.name
                        
                        tar.add(file_path, arcname=str(arcname))
                        total_size += file_path.stat().st_size
        
        else:  # No compression or GZIP
            # For simplicity, use tar.gz even for "no compression"
            with tarfile.open(archive_path, 'w') as tar:
                for file_path in files:
                    if file_path.exists():
                        try:
                            arcname = file_path.relative_to(self.memory_base_path)
                        except ValueError:
                            arcname = file_path.name
                        
                        tar.add(file_path, arcname=str(arcname))
                        total_size += file_path.stat().st_size
        
        return total_size
    
    def _extract_backup_archive(self, archive_path: Path, extract_dir: Path) -> List[Path]:
        """Extract backup archive and return list of extracted files."""
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = []
        
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(extract_dir)
            
            # Get list of extracted files
            for member in tar.getmembers():
                if member.isfile():
                    extracted_files.append(extract_dir / member.name)
        
        return extracted_files
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        if not file_path.exists():
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _verify_backup(self, backup_path: Path, original_files: List[Path]) -> bool:
        """Verify backup integrity."""
        try:
            # Extract to temporary directory
            temp_dir = self.backup_dir / f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            try:
                extracted_files = self._extract_backup_archive(backup_path, temp_dir)
                
                # Verify file count
                if len(extracted_files) != len(original_files):
                    logger.warning(f"File count mismatch: {len(extracted_files)} vs {len(original_files)}")
                    return False
                
                # Verify file contents (sample check)
                sample_size = min(10, len(extracted_files))
                for i in range(0, len(extracted_files), max(1, len(extracted_files) // sample_size)):
                    extracted_file = extracted_files[i]
                    
                    # Find corresponding original file
                    relative_path = extracted_file.relative_to(temp_dir)
                    original_file = self.memory_base_path / relative_path
                    
                    if original_file.exists():
                        if extracted_file.stat().st_size != original_file.stat().st_size:
                            logger.warning(f"Size mismatch for {relative_path}")
                            return False
                
                return True
                
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    def _verify_backup_integrity(self, backup_metadata: BackupMetadata) -> bool:
        """Verify backup file integrity using checksum."""
        backup_path = Path(backup_metadata.backup_path)
        if not backup_path.exists():
            return False
        
        current_checksum = self._calculate_file_checksum(backup_path)
        return current_checksum == backup_metadata.checksum
    
    def _verify_restored_files(
        self,
        extracted_files: List[Path],
        target_dir: Path,
        temp_dir: Path
    ) -> bool:
        """Verify that restored files match the extracted files."""
        try:
            for extracted_file in extracted_files:
                relative_path = extracted_file.relative_to(temp_dir)
                restored_file = target_dir / relative_path
                
                if not restored_file.exists():
                    logger.warning(f"Restored file missing: {restored_file}")
                    return False
                
                if extracted_file.stat().st_size != restored_file.stat().st_size:
                    logger.warning(f"Size mismatch for restored file: {restored_file}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Restore verification failed: {e}")
            return False
    
    def _update_file_tracking(self, files: List[Path], backup_id: str) -> None:
        """Update file tracking for incremental backups."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for file_path in files:
                    if file_path.exists():
                        file_path_str = str(file_path)
                        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        file_size = file_path.stat().st_size
                        checksum = self._calculate_file_checksum(file_path)
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO file_tracking 
                            (file_path, last_modified, last_backup_id, file_size, checksum)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (file_path_str, last_modified, backup_id, file_size, checksum))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating file tracking: {e}")
    
    def _record_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Record backup metadata in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Convert metadata to dict with enum values as strings
                metadata_dict = asdict(metadata)
                metadata_dict['backup_type'] = metadata.backup_type.value
                metadata_dict['status'] = metadata.status.value
                
                cursor.execute('''
                    INSERT OR REPLACE INTO backups 
                    (backup_id, backup_type, created_at, completed_at, status, file_count,
                     total_size_bytes, compressed_size_bytes, checksum, base_backup_id,
                     backup_path, compression_ratio, verification_status, error_message, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.backup_id, metadata.backup_type.value, metadata.created_at,
                    metadata.completed_at, metadata.status.value, metadata.file_count,
                    metadata.total_size_bytes, metadata.compressed_size_bytes, metadata.checksum,
                    metadata.base_backup_id, metadata.backup_path, metadata.compression_ratio,
                    metadata.verification_status, metadata.error_message, json.dumps(metadata_dict)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording backup metadata: {e}")
    
    def _save_backup_metadata_file(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to JSON file."""
        try:
            metadata_file = self.metadata_dir / f"{metadata.backup_id}.json"
            # Convert metadata to dict with enum values as strings
            metadata_dict = asdict(metadata)
            metadata_dict['backup_type'] = metadata.backup_type.value
            metadata_dict['status'] = metadata.status.value
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving backup metadata file: {e}")
    
    def _record_restore_operation(self, result: RestoreResult, options: RestoreOptions) -> None:
        """Record restore operation in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO restore_operations 
                    (restore_id, backup_id, started_at, completed_at, status, files_restored,
                     files_failed, total_size_restored, verification_passed, error_messages,
                     restore_log_path, options_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.restore_id, result.backup_id, result.started_at, result.completed_at,
                    result.status, result.files_restored, result.files_failed,
                    result.total_size_restored, result.verification_passed,
                    json.dumps(result.error_messages), result.restore_log_path,
                    json.dumps(asdict(options))
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording restore operation: {e}")
    
    def _get_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get backup metadata by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM backups WHERE backup_id = ?', (backup_id,))
                row = cursor.fetchone()
                
                if row:
                    return BackupMetadata(
                        backup_id=row[0],
                        backup_type=BackupType(row[1]),
                        created_at=row[2],
                        completed_at=row[3],
                        status=BackupStatus(row[4]),
                        file_count=row[5] or 0,
                        total_size_bytes=row[6] or 0,
                        compressed_size_bytes=row[7] or 0,
                        checksum=row[8] or "",
                        base_backup_id=row[9],
                        backup_path=row[10] or "",
                        compression_ratio=row[11] or 0.0,
                        verification_status=bool(row[12]) if row[12] is not None else False,
                        error_message=row[13],
                        files_included=[]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting backup metadata: {e}")
            return None
    
    def _get_latest_backup(self, backup_type: BackupType) -> Optional[BackupMetadata]:
        """Get the latest backup of specified type."""
        backups = self.list_backups(backup_type, limit=1)
        return backups[0] if backups else None
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy."""
        try:
            backups = self.list_backups(limit=1000)  # Get all backups
            
            if len(backups) > self.config.max_backups:
                # Sort by creation date (oldest first)
                backups.sort(key=lambda b: b.created_at)
                
                # Delete oldest backups
                backups_to_delete = backups[:-self.config.max_backups]
                
                for backup in backups_to_delete:
                    if backup.status == BackupStatus.COMPLETED:
                        self.delete_backup(backup.backup_id)
                        logger.info(f"Deleted old backup: {backup.backup_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def _auto_backup_worker(self) -> None:
        """Worker thread for automatic backups."""
        last_full_backup = time.time()
        last_incremental_backup = time.time()
        
        while self._auto_backup_running:
            try:
                current_time = time.time()
                
                # Check for full backup
                if (current_time - last_full_backup) >= (self.config.auto_backup_interval_hours * 3600):
                    logger.info("Creating scheduled full backup")
                    self.create_backup(BackupType.FULL)
                    last_full_backup = current_time
                
                # Check for incremental backup
                elif (current_time - last_incremental_backup) >= (self.config.incremental_backup_interval_hours * 3600):
                    logger.info("Creating scheduled incremental backup")
                    self.create_backup(BackupType.INCREMENTAL)
                    last_incremental_backup = current_time
                
                # Sleep for 1 hour before next check
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in auto-backup worker: {e}")
                time.sleep(3600)  # Continue after error


# Convenience functions for common backup operations
def create_emergency_backup(memory_path: str, backup_dir: str) -> Optional[BackupMetadata]:
    """Create an emergency backup quickly."""
    try:
        config = BackupConfig(
            backup_dir=backup_dir,
            compression=CompressionType.TAR_GZ,
            verify_backups=False  # Skip verification for speed
        )
        
        backup_system = MemoryBackupSystem(memory_path, config)
        return backup_system.create_backup(BackupType.SNAPSHOT)
        
    except Exception as e:
        logger.error(f"Emergency backup failed: {e}")
        return None


def restore_from_latest_backup(memory_path: str, backup_dir: str) -> Optional[RestoreResult]:
    """Restore from the latest available backup."""
    try:
        config = BackupConfig(backup_dir=backup_dir)
        backup_system = MemoryBackupSystem(memory_path, config)
        
        # Get latest backup
        backups = backup_system.list_backups(limit=1)
        if not backups:
            logger.error("No backups available for restore")
            return None
        
        latest_backup = backups[0]
        
        # Create restore options
        options = RestoreOptions(
            target_directory=memory_path,
            backup_id=latest_backup.backup_id,
            overwrite_existing=True
        )
        
        return backup_system.restore_backup(latest_backup.backup_id, options)
        
    except Exception as e:
        logger.error(f"Restore from latest backup failed: {e}")
        return None

