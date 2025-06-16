# Memory System Directory

This directory contains system configuration, operational data, and infrastructure files for the memory management system.

## Directory Structure

```
system/
├── README.md                    # This file
├── config.json                 # System configuration (tracked in Git)
├── analytics/                  # Analytics databases and reports (ignored)
├── backups/                    # Backup files and metadata (ignored)
├── embeddings/                 # Vector embeddings and indexes (ignored)
├── logs/                       # System operation logs (ignored)
└── temp/                       # Temporary files and cache (ignored)
```

## Configuration Files

### `config.json` (Tracked)
Main system configuration file that should be committed to Git:

```json
{
  "version": "1.0.0",
  "memory_settings": {
    "max_interactions_per_month": 10000,
    "importance_threshold": 5,
    "auto_cleanup_enabled": true,
    "backup_enabled": true
  },
  "paths": {
    "interactions_dir": "memory/interactions",
    "core_dir": "memory/core",
    "condensed_dir": "memory/condensed",
    "archive_dir": "memory/archive"
  },
  "features": {
    "analytics_enabled": true,
    "embeddings_enabled": true,
    "auto_backup": true,
    "compression": true
  }
}
```

## Operational Directories

### `analytics/` (Ignored)
Contains analytics databases and generated reports:
- `memory_analytics.db` - SQLite database with usage statistics
- `reports/` - Generated analytics reports
- `metrics/` - Performance and usage metrics

### `backups/` (Ignored)
Contains backup files and metadata:
- `backup_registry.db` - Backup metadata database
- `metadata/` - Individual backup metadata files
- `restore_logs/` - Restoration operation logs
- `*.tar.gz` - Compressed backup archives

### `embeddings/` (Ignored)
Contains vector embeddings and search indexes:
- `*.faiss` - FAISS vector indexes
- `*.chroma` - ChromaDB vector stores
- `embeddings_cache/` - Cached embedding vectors
- `search_indexes/` - Full-text search indexes

### `logs/` (Ignored)
Contains system operation logs:
- `memory_manager.log` - Memory system operations
- `backup_operations.log` - Backup and restore logs
- `analytics.log` - Analytics system logs
- `cleanup.log` - Cleanup operation logs

### `temp/` (Ignored)
Contains temporary files and cache:
- Processing temporary files
- Cache files for performance optimization
- Temporary extraction directories
- Work-in-progress files

## Privacy and Security

⚠️ **Data Protection**: Most system directories contain operational data that should not be committed to version control.

### Tracked Files
- `config.json` - System configuration (no sensitive data)
- `README.md` files - Documentation

### Ignored Files
- All databases (`.db`, `.sqlite`, `.sqlite3`)
- All backup archives and metadata
- All log files
- All cache and temporary files
- All vector embeddings and indexes

## File Management

### Automatic Cleanup
The system automatically manages:
- **Log Rotation**: Old logs are compressed and archived
- **Cache Cleanup**: Temporary files are regularly purged
- **Backup Retention**: Old backups are removed per retention policy
- **Index Maintenance**: Search indexes are rebuilt as needed

### Manual Maintenance
Administrators can:
- Clear all caches: Remove `temp/` contents
- Reset analytics: Delete `analytics/` directory
- Force backup: Manually trigger backup operations
- Rebuild indexes: Regenerate search and vector indexes

## Integration Points

### Memory Manager
- Reads configuration from `config.json`
- Writes operational logs to `logs/`
- Uses `temp/` for processing

### Analytics System
- Stores data in `analytics/memory_analytics.db`
- Generates reports in `analytics/reports/`
- Tracks metrics in `analytics/metrics/`

### Backup System
- Maintains registry in `backups/backup_registry.db`
- Stores archives in `backups/`
- Logs operations to `backups/restore_logs/`

### Search System
- Maintains indexes in `embeddings/`
- Caches results in `temp/`
- Logs operations to `logs/`

## Configuration Management

### Environment-Specific Settings
- Development: Use local paths and verbose logging
- Production: Use optimized paths and error-only logging
- Testing: Use temporary directories and mock services

### Feature Toggles
Configuration supports enabling/disabling:
- Analytics collection
- Vector embeddings
- Automatic backups
- Memory compression
- Advanced search features

### Performance Tuning
Adjustable parameters:
- Cache sizes and timeouts
- Backup frequency and retention
- Analytics sampling rates
- Cleanup thresholds and schedules 