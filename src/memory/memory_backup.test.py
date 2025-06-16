"""
Unit tests for Memory Backup and Recovery System

Tests for comprehensive backup functionality including backup creation,
restoration, incremental backups, compression, and auto-backup features.
"""

import pytest
import tempfile
import json
import sqlite3
import tarfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import threading
import time

# Handle imports gracefully for both package and standalone execution
try:
    from .memory_backup import (
        MemoryBackupSystem, BackupType, BackupStatus, CompressionType,
        BackupConfig, BackupMetadata, RestoreOptions, RestoreResult,
        create_emergency_backup, restore_from_latest_backup
    )
    from .file_operations import write_memory_file
except ImportError:
    from memory_backup import (
        MemoryBackupSystem, BackupType, BackupStatus, CompressionType,
        BackupConfig, BackupMetadata, RestoreOptions, RestoreResult,
        create_emergency_backup, restore_from_latest_backup
    )
    from file_operations import write_memory_file


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create a temporary memory directory for testing"""
    memory_dir = tmp_path / "test_memory"
    memory_dir.mkdir()
    
    # Create required subdirectories
    (memory_dir / "interactions").mkdir()
    (memory_dir / "core").mkdir()
    (memory_dir / "system").mkdir()
    
    return str(memory_dir)


@pytest.fixture
def backup_config(tmp_path):
    """Create a backup configuration for testing"""
    backup_dir = tmp_path / "backups"
    return BackupConfig(
        backup_dir=str(backup_dir),
        max_backups=5,
        compression=CompressionType.TAR_GZ,
        verify_backups=True,
        auto_backup_interval_hours=1,
        incremental_backup_interval_hours=1
    )


@pytest.fixture
def backup_system(temp_memory_dir, backup_config):
    """Create a MemoryBackupSystem instance for testing"""
    return MemoryBackupSystem(temp_memory_dir, backup_config)


@pytest.fixture
def sample_memory_files(temp_memory_dir):
    """Create sample memory files for backup testing"""
    base_path = Path(temp_memory_dir)
    files_created = {}
    
    # Create sample files in different directories
    
    # Interaction file
    interaction_file = base_path / "interactions" / "test_interaction.md"
    interaction_frontmatter = {
        'created': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'importance_score': 8,
        'memory_type': 'interaction',
        'category': 'work'
    }
    interaction_content = "Test interaction content for backup testing."
    write_memory_file(interaction_file, interaction_frontmatter, interaction_content)
    files_created['interaction'] = str(interaction_file)
    
    # Core file
    core_file = base_path / "core" / "user_profile.md"
    core_frontmatter = {
        'created': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'importance_score': 9,
        'memory_type': 'user_profile',
        'category': 'core'
    }
    core_content = "User profile information for backup testing."
    write_memory_file(core_file, core_frontmatter, core_content)
    files_created['core'] = str(core_file)
    
    # System file
    system_file = base_path / "system" / "config.json"
    system_data = {
        'version': '1.0',
        'settings': {
            'backup_enabled': True,
            'compression': 'gzip'
        }
    }
    with open(system_file, 'w') as f:
        json.dump(system_data, f, indent=2)
    files_created['system'] = str(system_file)
    
    return files_created


class TestBackupSystemInitialization:
    """Test cases for MemoryBackupSystem initialization"""
    
    def test_initialization_basic(self, temp_memory_dir, backup_config):
        """Test basic initialization"""
        backup_system = MemoryBackupSystem(temp_memory_dir, backup_config)
        
        assert backup_system.memory_base_path == Path(temp_memory_dir)
        assert backup_system.config == backup_config
        assert backup_system.backup_dir.exists()
        assert backup_system.metadata_dir.exists()
        assert backup_system.restore_logs_dir.exists()
        assert backup_system.db_path.exists()
        assert not backup_system._auto_backup_running
    
    def test_initialization_default_config(self, temp_memory_dir):
        """Test initialization with default configuration"""
        backup_system = MemoryBackupSystem(temp_memory_dir)
        
        assert backup_system.config.max_backups == 10
        assert backup_system.config.compression == CompressionType.TAR_GZ
        assert backup_system.config.verify_backups == True
        assert backup_system.config.auto_backup_interval_hours == 24
    
    def test_database_initialization(self, backup_system):
        """Test that backup database is properly initialized"""
        # Check that database file exists
        assert backup_system.db_path.exists()
        
        # Check that tables are created
        with sqlite3.connect(backup_system.db_path) as conn:
            cursor = conn.cursor()
            
            # Check backups table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='backups'")
            assert cursor.fetchone() is not None
            
            # Check restore_operations table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='restore_operations'")
            assert cursor.fetchone() is not None
            
            # Check file_tracking table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_tracking'")
            assert cursor.fetchone() is not None
    
    def test_directory_structure_creation(self, temp_memory_dir, backup_config):
        """Test that required directories are created"""
        backup_system = MemoryBackupSystem(temp_memory_dir, backup_config)
        
        expected_backup_dir = Path(backup_config.backup_dir)
        expected_metadata_dir = expected_backup_dir / 'metadata'
        expected_restore_logs_dir = expected_backup_dir / 'restore_logs'
        
        assert expected_backup_dir.exists()
        assert expected_metadata_dir.exists()
        assert expected_restore_logs_dir.exists()


class TestBackupCreation:
    """Test cases for backup creation"""
    
    def test_create_full_backup_basic(self, backup_system, sample_memory_files):
        """Test basic full backup creation"""
        metadata = backup_system.create_backup(BackupType.FULL)
        
        assert isinstance(metadata, BackupMetadata)
        assert metadata.backup_type == BackupType.FULL
        assert metadata.status == BackupStatus.COMPLETED
        assert metadata.file_count > 0
        assert metadata.total_size_bytes > 0
        assert metadata.compressed_size_bytes > 0
        assert metadata.compression_ratio >= 0
        assert metadata.checksum != ""
        assert metadata.verification_status == True
        assert Path(metadata.backup_path).exists()
    
    def test_create_snapshot_backup(self, backup_system, sample_memory_files):
        """Test snapshot backup creation"""
        metadata = backup_system.create_backup(BackupType.SNAPSHOT)
        
        assert metadata.backup_type == BackupType.SNAPSHOT
        assert metadata.status == BackupStatus.COMPLETED
        assert metadata.base_backup_id is None
        assert Path(metadata.backup_path).exists()
    
    def test_create_incremental_backup_without_base(self, backup_system, sample_memory_files):
        """Test incremental backup creation without base backup (should create full backup)"""
        metadata = backup_system.create_backup(BackupType.INCREMENTAL)
        
        # Should fallback to full backup
        assert metadata.backup_type == BackupType.FULL
        assert metadata.status == BackupStatus.COMPLETED
        assert metadata.base_backup_id is None
    
    def test_create_incremental_backup_with_base(self, backup_system, sample_memory_files):
        """Test incremental backup creation with base backup"""
        # Create full backup first
        full_backup = backup_system.create_backup(BackupType.FULL)
        
        # Modify a file
        base_path = Path(backup_system.memory_base_path)
        test_file = base_path / "interactions" / "new_file.md"
        write_memory_file(test_file, {'created': datetime.now().isoformat()}, "New content")
        
        # Create incremental backup
        incremental_backup = backup_system.create_backup(BackupType.INCREMENTAL)
        
        assert incremental_backup.backup_type == BackupType.INCREMENTAL
        assert incremental_backup.status == BackupStatus.COMPLETED
        assert incremental_backup.base_backup_id == full_backup.backup_id
        assert incremental_backup.file_count >= 1  # At least the new file
    
    def test_backup_compression_tar_gz(self, temp_memory_dir, sample_memory_files):
        """Test backup with TAR_GZ compression"""
        config = BackupConfig(
            backup_dir=str(Path(temp_memory_dir) / "backups"),
            compression=CompressionType.TAR_GZ
        )
        backup_system = MemoryBackupSystem(temp_memory_dir, config)
        
        metadata = backup_system.create_backup(BackupType.FULL)
        
        assert metadata.status == BackupStatus.COMPLETED
        assert metadata.backup_path.endswith('.tar.gz')
        
        # Verify it's a valid tar.gz file
        backup_path = Path(metadata.backup_path)
        assert tarfile.is_tarfile(backup_path)
    
    def test_backup_compression_tar_bz2(self, temp_memory_dir, sample_memory_files):
        """Test backup with TAR_BZ2 compression"""
        config = BackupConfig(
            backup_dir=str(Path(temp_memory_dir) / "backups"),
            compression=CompressionType.TAR_BZ2
        )
        backup_system = MemoryBackupSystem(temp_memory_dir, config)
        
        metadata = backup_system.create_backup(BackupType.FULL)
        
        assert metadata.status == BackupStatus.COMPLETED
        assert metadata.backup_path.endswith('.tar.bz2')
        assert tarfile.is_tarfile(metadata.backup_path)
    
    def test_backup_verification_enabled(self, backup_system, sample_memory_files):
        """Test backup with verification enabled"""
        backup_system.config.verify_backups = True
        
        metadata = backup_system.create_backup(BackupType.FULL)
        
        assert metadata.verification_status == True
        assert metadata.status == BackupStatus.COMPLETED
    
    def test_backup_verification_disabled(self, backup_system, sample_memory_files):
        """Test backup with verification disabled"""
        backup_system.config.verify_backups = False
        
        metadata = backup_system.create_backup(BackupType.FULL)
        
        assert metadata.verification_status == True  # Always set to True when disabled
        assert metadata.status == BackupStatus.COMPLETED
    
    def test_backup_metadata_persistence(self, backup_system, sample_memory_files):
        """Test that backup metadata is properly persisted"""
        metadata = backup_system.create_backup(BackupType.FULL)
        
        # Check database record
        with sqlite3.connect(backup_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM backups WHERE backup_id = ?', (metadata.backup_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == metadata.backup_id
            assert row[1] == metadata.backup_type.value
        
        # Check metadata file
        metadata_file = backup_system.metadata_dir / f"{metadata.backup_id}.json"
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata['backup_id'] == metadata.backup_id
        assert saved_metadata['backup_type'] == metadata.backup_type.value


class TestBackupListing:
    """Test cases for backup listing"""
    
    def test_list_backups_empty(self, backup_system):
        """Test listing backups when none exist"""
        backups = backup_system.list_backups()
        
        assert backups == []
    
    def test_list_backups_basic(self, backup_system, sample_memory_files):
        """Test basic backup listing"""
        # Create multiple backups
        backup1 = backup_system.create_backup(BackupType.FULL)
        backup2 = backup_system.create_backup(BackupType.SNAPSHOT)
        
        backups = backup_system.list_backups()
        
        assert len(backups) == 2
        assert backups[0].backup_id == backup2.backup_id  # Most recent first
        assert backups[1].backup_id == backup1.backup_id
    
    def test_list_backups_by_type(self, backup_system, sample_memory_files):
        """Test listing backups filtered by type"""
        backup_system.create_backup(BackupType.FULL)
        backup_system.create_backup(BackupType.SNAPSHOT)
        backup_system.create_backup(BackupType.SNAPSHOT)
        
        full_backups = backup_system.list_backups(BackupType.FULL)
        snapshot_backups = backup_system.list_backups(BackupType.SNAPSHOT)
        
        assert len(full_backups) == 1
        assert len(snapshot_backups) == 2
        assert all(b.backup_type == BackupType.FULL for b in full_backups)
        assert all(b.backup_type == BackupType.SNAPSHOT for b in snapshot_backups)
    
    def test_list_backups_with_limit(self, backup_system, sample_memory_files):
        """Test listing backups with limit"""
        # Create multiple backups
        for i in range(5):
            backup_system.create_backup(BackupType.SNAPSHOT)
        
        backups = backup_system.list_backups(limit=3)
        
        assert len(backups) == 3


class TestBackupDeletion:
    """Test cases for backup deletion"""
    
    def test_delete_backup_basic(self, backup_system, sample_memory_files):
        """Test basic backup deletion"""
        metadata = backup_system.create_backup(BackupType.FULL)
        backup_path = Path(metadata.backup_path)
        metadata_file = backup_system.metadata_dir / f"{metadata.backup_id}.json"
        
        # Verify files exist
        assert backup_path.exists()
        assert metadata_file.exists()
        
        # Delete backup
        result = backup_system.delete_backup(metadata.backup_id)
        
        assert result == True
        assert not backup_path.exists()
        assert not metadata_file.exists()
        
        # Verify database record is removed
        with sqlite3.connect(backup_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM backups WHERE backup_id = ?', (metadata.backup_id,))
            assert cursor.fetchone() is None
    
    def test_delete_nonexistent_backup(self, backup_system):
        """Test deleting a non-existent backup"""
        result = backup_system.delete_backup("nonexistent_backup")
        
        assert result == False
    
    def test_cleanup_old_backups(self, backup_system, sample_memory_files):
        """Test automatic cleanup of old backups"""
        backup_system.config.max_backups = 3
        
        # Create more backups than the limit
        backup_ids = []
        for i in range(5):
            metadata = backup_system.create_backup(BackupType.SNAPSHOT)
            backup_ids.append(metadata.backup_id)
        
        # List remaining backups
        remaining_backups = backup_system.list_backups()
        
        # Should only have max_backups (3) remaining
        assert len(remaining_backups) == 3
        
        # Should be the most recent ones
        remaining_ids = [b.backup_id for b in remaining_backups]
        assert backup_ids[-3:] == remaining_ids[::-1]  # Most recent first in list


class TestBackupRestore:
    """Test cases for backup restoration"""
    
    def test_restore_backup_basic(self, backup_system, sample_memory_files, tmp_path):
        """Test basic backup restoration"""
        # Create backup
        metadata = backup_system.create_backup(BackupType.FULL)
        
        # Create restore target directory
        restore_dir = tmp_path / "restore_target"
        
        # Create restore options
        options = RestoreOptions(
            target_directory=str(restore_dir),
            backup_id=metadata.backup_id,
            overwrite_existing=True
        )
        
        # Restore backup
        result = backup_system.restore_backup(metadata.backup_id, options)
        
        assert isinstance(result, RestoreResult)
        assert result.status == "completed"
        assert result.files_restored > 0
        assert result.files_failed == 0
        assert result.verification_passed == True
        assert restore_dir.exists()
        
        # Verify files were restored
        assert (restore_dir / "interactions" / "test_interaction.md").exists()
        assert (restore_dir / "core" / "user_profile.md").exists()
        assert (restore_dir / "system" / "config.json").exists()
    
    def test_restore_backup_selective(self, backup_system, sample_memory_files, tmp_path):
        """Test selective backup restoration"""
        metadata = backup_system.create_backup(BackupType.FULL)
        restore_dir = tmp_path / "restore_target"
        
        # Select only specific files
        selected_files = ["interactions/test_interaction.md"]
        
        options = RestoreOptions(
            target_directory=str(restore_dir),
            backup_id=metadata.backup_id,
            restore_type="selective",
            selected_files=selected_files,
            overwrite_existing=True
        )
        
        result = backup_system.restore_backup(metadata.backup_id, options)
        
        assert result.status == "completed"
        assert result.files_restored >= 1
        
        # Only selected files should be restored
        assert (restore_dir / "interactions" / "test_interaction.md").exists()
        # Other files should not be restored
        assert not (restore_dir / "core" / "user_profile.md").exists()
    
    def test_restore_backup_without_overwrite(self, backup_system, sample_memory_files, tmp_path):
        """Test restoration without overwriting existing files"""
        metadata = backup_system.create_backup(BackupType.FULL)
        restore_dir = tmp_path / "restore_target"
        restore_dir.mkdir(parents=True)
        
        # Create existing file
        existing_file = restore_dir / "interactions" / "test_interaction.md"
        existing_file.parent.mkdir(parents=True)
        existing_file.write_text("Existing content")
        
        options = RestoreOptions(
            target_directory=str(restore_dir),
            backup_id=metadata.backup_id,
            overwrite_existing=False
        )
        
        result = backup_system.restore_backup(metadata.backup_id, options)
        
        assert result.status == "completed"
        
        # Existing file should not be overwritten
        assert existing_file.read_text() == "Existing content"
    
    def test_restore_backup_with_verification(self, backup_system, sample_memory_files, tmp_path):
        """Test restoration with integrity verification"""
        metadata = backup_system.create_backup(BackupType.FULL)
        restore_dir = tmp_path / "restore_target"
        
        options = RestoreOptions(
            target_directory=str(restore_dir),
            backup_id=metadata.backup_id,
            verify_integrity=True
        )
        
        result = backup_system.restore_backup(metadata.backup_id, options)
        
        assert result.verification_passed == True
        assert result.status == "completed"
    
    def test_restore_nonexistent_backup(self, backup_system, tmp_path):
        """Test restoring from non-existent backup"""
        restore_dir = tmp_path / "restore_target"
        
        options = RestoreOptions(
            target_directory=str(restore_dir),
            backup_id="nonexistent_backup"
        )
        
        with pytest.raises(ValueError, match="Backup nonexistent_backup not found"):
            backup_system.restore_backup("nonexistent_backup", options)
    
    def test_restore_with_log_creation(self, backup_system, sample_memory_files, tmp_path):
        """Test restoration with log file creation"""
        metadata = backup_system.create_backup(BackupType.FULL)
        restore_dir = tmp_path / "restore_target"
        
        options = RestoreOptions(
            target_directory=str(restore_dir),
            backup_id=metadata.backup_id,
            create_restore_log=True
        )
        
        result = backup_system.restore_backup(metadata.backup_id, options)
        
        assert result.restore_log_path is not None
        log_file = Path(result.restore_log_path)
        assert log_file.exists()
        
        # Verify log content
        log_content = log_file.read_text()
        assert "RESTORED:" in log_content


class TestAutoBackup:
    """Test cases for automatic backup functionality"""
    
    def test_start_stop_auto_backup(self, backup_system):
        """Test starting and stopping auto-backup"""
        assert not backup_system._auto_backup_running
        
        backup_system.start_auto_backup()
        assert backup_system._auto_backup_running
        assert backup_system._auto_backup_thread is not None
        
        backup_system.stop_auto_backup()
        assert not backup_system._auto_backup_running
    
    def test_start_auto_backup_already_running(self, backup_system):
        """Test starting auto-backup when already running"""
        backup_system.start_auto_backup()
        
        # Try to start again
        backup_system.start_auto_backup()  # Should not raise error
        
        assert backup_system._auto_backup_running
        
        backup_system.stop_auto_backup()
    
    @patch('time.sleep')
    def test_auto_backup_worker_logic(self, mock_sleep, backup_system, sample_memory_files):
        """Test auto-backup worker logic (mocked)"""
        # Set very short intervals for testing
        backup_system.config.auto_backup_interval_hours = 0.001  # ~3.6 seconds
        backup_system.config.incremental_backup_interval_hours = 0.0005  # ~1.8 seconds
        
        # Mock time.time to simulate passage of time
        start_time = time.time()
        time_values = [
            start_time,  # Initial time
            start_time + 7200,  # 2 hours later (trigger incremental)
            start_time + 14400,  # 4 hours later (trigger full)
        ]
        
        with patch('time.time', side_effect=time_values):
            with patch.object(backup_system, 'create_backup') as mock_create_backup:
                # Start auto-backup
                backup_system._auto_backup_running = True
                
                # Run one iteration of the worker
                backup_system._auto_backup_worker()
                
                # Should have created backups
                assert mock_create_backup.call_count >= 1


class TestBackupStatistics:
    """Test cases for backup statistics"""
    
    def test_get_backup_statistics_empty(self, backup_system):
        """Test backup statistics with no backups"""
        stats = backup_system.get_backup_statistics()
        
        assert stats['total_backups'] == 0
        assert stats['backups_by_type'] == {}
        assert stats['total_backup_size_bytes'] == 0
        assert stats['failed_backups'] == 0
        assert stats['total_restore_operations'] == 0
        assert stats['auto_backup_running'] == False
    
    def test_get_backup_statistics_with_data(self, backup_system, sample_memory_files, tmp_path):
        """Test backup statistics with actual data"""
        # Create various backups
        backup_system.create_backup(BackupType.FULL)
        backup_system.create_backup(BackupType.SNAPSHOT)
        backup_system.create_backup(BackupType.SNAPSHOT)
        
        # Perform a restore
        backups = backup_system.list_backups(limit=1)
        restore_dir = tmp_path / "restore_test"
        options = RestoreOptions(
            target_directory=str(restore_dir),
            backup_id=backups[0].backup_id
        )
        backup_system.restore_backup(backups[0].backup_id, options)
        
        stats = backup_system.get_backup_statistics()
        
        assert stats['total_backups'] == 3
        assert stats['backups_by_type']['full'] == 1
        assert stats['backups_by_type']['snapshot'] == 2
        assert stats['total_backup_size_bytes'] > 0
        assert stats['average_compression_ratio'] >= 0
        assert stats['total_restore_operations'] == 1
        assert 'backup_directory' in stats
        assert 'config' in stats


class TestBackupIntegrity:
    """Test cases for backup integrity and verification"""
    
    def test_backup_checksum_calculation(self, backup_system, sample_memory_files):
        """Test that backup checksums are calculated correctly"""
        metadata = backup_system.create_backup(BackupType.FULL)
        
        # Verify checksum is not empty
        assert metadata.checksum != ""
        assert len(metadata.checksum) == 64  # SHA-256 hex length
        
        # Verify checksum matches file
        backup_path = Path(metadata.backup_path)
        calculated_checksum = backup_system._calculate_file_checksum(backup_path)
        assert calculated_checksum == metadata.checksum
    
    def test_backup_integrity_verification(self, backup_system, sample_memory_files):
        """Test backup integrity verification"""
        metadata = backup_system.create_backup(BackupType.FULL)
        
        # Verify integrity
        is_valid = backup_system._verify_backup_integrity(metadata)
        assert is_valid == True
        
        # Corrupt the backup file
        backup_path = Path(metadata.backup_path)
        with open(backup_path, 'ab') as f:
            f.write(b'corrupted_data')
        
        # Verification should fail
        is_valid = backup_system._verify_backup_integrity(metadata)
        assert is_valid == False
    
    def test_backup_content_verification(self, backup_system, sample_memory_files):
        """Test backup content verification during creation"""
        backup_system.config.verify_backups = True
        
        metadata = backup_system.create_backup(BackupType.FULL)
        
        assert metadata.verification_status == True
        assert metadata.status == BackupStatus.COMPLETED


class TestConvenienceFunctions:
    """Test cases for convenience functions"""
    
    def test_create_emergency_backup(self, temp_memory_dir, sample_memory_files, tmp_path):
        """Test emergency backup creation"""
        backup_dir = str(tmp_path / "emergency_backups")
        
        metadata = create_emergency_backup(temp_memory_dir, backup_dir)
        
        assert metadata is not None
        assert metadata.backup_type == BackupType.SNAPSHOT
        assert metadata.status == BackupStatus.COMPLETED
        assert Path(metadata.backup_path).exists()
    
    def test_restore_from_latest_backup(self, temp_memory_dir, sample_memory_files, tmp_path):
        """Test restore from latest backup"""
        backup_dir = str(tmp_path / "test_backups")
        
        # Create a backup first
        emergency_metadata = create_emergency_backup(temp_memory_dir, backup_dir)
        assert emergency_metadata is not None
        
        # Restore from latest backup
        result = restore_from_latest_backup(temp_memory_dir, backup_dir)
        
        assert result is not None
        assert result.status == "completed"
        assert result.files_restored > 0
    
    def test_restore_from_latest_backup_no_backups(self, temp_memory_dir, tmp_path):
        """Test restore from latest backup when no backups exist"""
        backup_dir = str(tmp_path / "empty_backups")
        
        result = restore_from_latest_backup(temp_memory_dir, backup_dir)
        
        assert result is None


class TestErrorHandling:
    """Test cases for error handling in backup system"""
    
    def test_backup_creation_error_handling(self, backup_system):
        """Test error handling during backup creation"""
        # Remove memory directory to cause error
        import shutil
        shutil.rmtree(backup_system.memory_base_path)
        
        with pytest.raises(Exception):
            backup_system.create_backup(BackupType.FULL)
    
    def test_restore_error_handling(self, backup_system, sample_memory_files):
        """Test error handling during restore"""
        metadata = backup_system.create_backup(BackupType.FULL)
        
        # Delete backup file to cause error
        backup_path = Path(metadata.backup_path)
        backup_path.unlink()
        
        options = RestoreOptions(
            target_directory="/tmp/restore_test",
            backup_id=metadata.backup_id
        )
        
        with pytest.raises(Exception):
            backup_system.restore_backup(metadata.backup_id, options)
    
    def test_database_error_handling(self, backup_system):
        """Test error handling with database issues"""
        # Corrupt the database
        backup_system.db_path.unlink()
        
        # Should handle gracefully
        backups = backup_system.list_backups()
        assert backups == []


class TestFileTracking:
    """Test cases for file tracking for incremental backups"""
    
    def test_file_tracking_update(self, backup_system, sample_memory_files):
        """Test file tracking updates"""
        # Create full backup
        metadata = backup_system.create_backup(BackupType.FULL)
        
        # Check file tracking was updated
        with sqlite3.connect(backup_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM file_tracking WHERE last_backup_id = ?', (metadata.backup_id,))
            count = cursor.fetchone()[0]
            
            assert count > 0
    
    def test_changed_files_detection(self, backup_system, sample_memory_files):
        """Test detection of changed files for incremental backup"""
        # Create full backup
        full_backup = backup_system.create_backup(BackupType.FULL)
        
        # Modify a file
        base_path = Path(backup_system.memory_base_path)
        test_file = base_path / "interactions" / "test_interaction.md"
        
        # Wait a moment to ensure different timestamp
        time.sleep(0.1)
        
        # Modify the file
        with open(test_file, 'a') as f:
            f.write("\nModified content")
        
        # Get changed files
        changed_files = backup_system._get_changed_files_since_backup(full_backup.backup_id)
        
        assert len(changed_files) >= 1
        assert test_file in changed_files


class TestCompressionTypes:
    """Test cases for different compression types"""
    
    def test_compression_none(self, temp_memory_dir, sample_memory_files):
        """Test backup with no compression"""
        config = BackupConfig(
            backup_dir=str(Path(temp_memory_dir) / "backups"),
            compression=CompressionType.NONE
        )
        backup_system = MemoryBackupSystem(temp_memory_dir, config)
        
        metadata = backup_system.create_backup(BackupType.FULL)
        
        assert metadata.status == BackupStatus.COMPLETED
        assert Path(metadata.backup_path).exists()
    
    def test_compression_ratio_calculation(self, backup_system, sample_memory_files):
        """Test compression ratio calculation"""
        metadata = backup_system.create_backup(BackupType.FULL)
        
        assert 0 <= metadata.compression_ratio <= 1
        assert metadata.total_size_bytes > 0
        assert metadata.compressed_size_bytes > 0
        assert metadata.compressed_size_bytes <= metadata.total_size_bytes


if __name__ == "__main__":
    pytest.main([__file__])