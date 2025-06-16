#!/usr/bin/env python3
"""
Test Script for Memory Validation and Error Handling System

This script demonstrates the functionality of the memory validation and error handling
system with real-world examples and edge cases.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from memory.validation import (
        ValidationSeverity,
        ErrorType,
        ValidationIssue,
        ValidationResult,
        MemoryValidator,
        MemoryErrorHandler,
        validate_memory_system
    )
    from memory.memory_initializer import MemoryInitializer
    from memory.file_operations import write_memory_file
    from memory.importance_scoring import TimestampManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the personal-ai-assistant directory")
    sys.exit(1)


def create_test_memories(memory_dir: Path):
    """Create various test memory files for validation testing."""
    
    print("Creating test memory files...")
    
    interactions_dir = memory_dir / 'interactions'
    core_dir = memory_dir / 'core'
    condensed_dir = memory_dir / 'condensed'
    
    # 1. Valid memory file
    valid_frontmatter = {
        'created': TimestampManager.create_timestamp(),
        'memory_type': 'interaction',
        'importance_score': 7,
        'categories': ['conversation', 'learning'],
        'tags': ['helpful', 'informative']
    }
    write_memory_file(
        interactions_dir / 'valid_memory.md',
        valid_frontmatter,
        "This is a perfectly valid memory file with proper structure and content."
    )
    
    # 2. Missing required fields
    invalid_frontmatter = {
        'created': TimestampManager.create_timestamp(),
        # Missing memory_type and importance_score
        'categories': ['incomplete']
    }
    write_memory_file(
        interactions_dir / 'missing_fields.md',
        invalid_frontmatter,
        "This memory file is missing required fields."
    )
    
    # 3. Invalid field types and values
    bad_types_frontmatter = {
        'created': TimestampManager.create_timestamp(),
        'memory_type': 123,  # Should be string
        'importance_score': "very high",  # Should be int
        'access_count': -5,  # Should be >= 0
        'categories': "not a list",  # Should be list
    }
    write_memory_file(
        interactions_dir / 'bad_types.md',
        bad_types_frontmatter,
        "This memory has incorrect field types."
    )
    
    # 4. Invalid timestamp formats
    bad_time_frontmatter = {
        'created': "not a timestamp",
        'memory_type': 'interaction',
        'importance_score': 5,
        'last_updated': "2024-13-45T25:99:99"  # Invalid datetime
    }
    write_memory_file(
        interactions_dir / 'bad_timestamps.md',
        bad_time_frontmatter,
        "This memory has invalid timestamp formats."
    )
    
    # 5. Core memory with low importance (should warn)
    low_importance_core = {
        'created': TimestampManager.create_timestamp(),
        'memory_type': 'user_profile',
        'importance_score': 2  # Too low for core memory
    }
    write_memory_file(
        core_dir / 'low_importance_profile.md',
        low_importance_core,
        "User profile with unusually low importance score."
    )
    
    # 6. Condensed memory with low source count (should warn)
    low_source_condensed = {
        'created': TimestampManager.create_timestamp(),
        'memory_type': 'condensed',
        'importance_score': 6,
        'source_count': 1  # Should be >= 2
    }
    write_memory_file(
        condensed_dir / 'low_source_count.md',
        low_source_condensed,
        "Condensed memory with only one source."
    )
    
    # 7. Memory with very long content (should warn)
    long_content_frontmatter = {
        'created': TimestampManager.create_timestamp(),
        'memory_type': 'interaction',
        'importance_score': 4
    }
    long_content = "This is a very long memory content. " * 10000  # ~370KB
    write_memory_file(
        interactions_dir / 'very_long_content.md',
        long_content_frontmatter,
        long_content
    )
    
    # 8. Empty content memory (should warn)
    empty_content_frontmatter = {
        'created': TimestampManager.create_timestamp(),
        'memory_type': 'interaction',
        'importance_score': 3
    }
    write_memory_file(
        interactions_dir / 'empty_content.md',
        empty_content_frontmatter,
        ""  # Empty content
    )
    
    # 9. Duplicate memory IDs (should error)
    duplicate_id_1 = {
        'created': TimestampManager.create_timestamp(),
        'memory_type': 'interaction',
        'importance_score': 5,
        'memory_id': 'duplicate_test_id'
    }
    write_memory_file(
        interactions_dir / 'duplicate_1.md',
        duplicate_id_1,
        "First memory with duplicate ID."
    )
    
    duplicate_id_2 = {
        'created': TimestampManager.create_timestamp(),
        'memory_type': 'interaction',
        'importance_score': 6,
        'memory_id': 'duplicate_test_id'  # Same ID!
    }
    write_memory_file(
        interactions_dir / 'duplicate_2.md',
        duplicate_id_2,
        "Second memory with duplicate ID."
    )
    
    print("✅ Created 9 test memory files with various validation issues")


def create_corrupted_file(memory_dir: Path):
    """Create a corrupted memory file for error handling testing."""
    
    corrupted_file = memory_dir / 'interactions' / 'corrupted.md'
    
    # Create corrupted YAML content
    corrupted_content = """---
created: 2024-01-01T10:00:00
memory_type: interaction
importance_score: 5
broken_yaml: [unclosed bracket and {invalid syntax
malformed_list:
  - item1
  - item2
  unclosed_item
---
This is the content part that should be recoverable.
The YAML frontmatter above is completely broken.
"""
    
    with open(corrupted_file, 'w', encoding='utf-8') as f:
        f.write(corrupted_content)
    
    print("✅ Created corrupted memory file for error handling testing")
    return corrupted_file


def demonstrate_validation():
    """Demonstrate the validation system functionality."""
    
    print("\n" + "="*60)
    print("MEMORY VALIDATION AND ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    # Create temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="memory_validation_test_"))
    print(f"\n📁 Using test directory: {test_dir}")
    
    try:
        # Initialize memory structure
        print("\n1. Initializing memory structure...")
        initializer = MemoryInitializer(str(test_dir))
        initializer.initialize_memory_structure()
        print("✅ Memory structure initialized")
        
        # Create test memories
        print("\n2. Creating test memory files...")
        create_test_memories(test_dir)
        
        # Create corrupted file
        print("\n3. Creating corrupted file...")
        corrupted_file = create_corrupted_file(test_dir)
        
        # Initialize validator and error handler
        validator = MemoryValidator(test_dir)
        error_handler = MemoryErrorHandler(test_dir)
        
        print("\n4. Running comprehensive memory system validation...")
        print("-" * 50)
        
        # Validate entire memory system
        result = validator.validate_memory_system()
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   {result.get_summary()}")
        print(f"   Total issues found: {len(result.issues)}")
        print(f"   Errors: {result.errors_count}")
        print(f"   Warnings: {result.warnings_count}")
        
        # Show detailed issues
        if result.issues:
            print(f"\n📋 DETAILED ISSUES:")
            for i, issue in enumerate(result.issues[:10], 1):  # Show first 10
                print(f"   {i}. {issue}")
                if issue.suggestion:
                    print(f"      💡 Suggestion: {issue.suggestion}")
            
            if len(result.issues) > 10:
                print(f"   ... and {len(result.issues) - 10} more issues")
        
        # Demonstrate error handling and repair
        print(f"\n5. Demonstrating error handling and repair...")
        print("-" * 50)
        
        # Try to repair the corrupted file
        print(f"\n🔧 Attempting to repair corrupted file: {corrupted_file.name}")
        repair_success = error_handler.repair_corrupted_file(corrupted_file)
        
        if repair_success:
            print("✅ File repair successful!")
            
            # Validate the repaired file
            repaired_result = validator.validate_memory_file(corrupted_file)
            print(f"   Repaired file validation: {repaired_result.get_summary()}")
        else:
            print("❌ File repair failed")
        
        # Demonstrate backup and restore
        print(f"\n💾 Demonstrating backup and restore functionality...")
        
        # Create a backup of a valid file
        valid_file = test_dir / 'interactions' / 'valid_memory.md'
        backup_path = error_handler.create_backup(valid_file)
        
        if backup_path:
            print(f"✅ Created backup: {backup_path.name}")
            
            # Simulate file corruption
            with open(valid_file, 'w', encoding='utf-8') as f:
                f.write("CORRUPTED CONTENT")
            
            print("⚠️  Simulated file corruption")
            
            # Restore from backup
            restore_success = error_handler.restore_from_backup(valid_file, backup_path)
            if restore_success:
                print("✅ File restored successfully from backup")
            else:
                print("❌ File restoration failed")
        
        # Demonstrate auto-fix functionality
        print(f"\n🔄 Demonstrating auto-fix functionality...")
        
        # Create a file with fixable issues
        fixable_file = test_dir / 'interactions' / 'fixable.md'
        fixable_frontmatter = {
            'created': TimestampManager.create_timestamp()
            # Missing memory_type and importance_score (auto-fixable)
        }
        write_memory_file(fixable_file, fixable_frontmatter, "Content that needs fixing")
        
        # Validate with auto-fix
        print("   Running validation with auto-fix enabled...")
        auto_fix_result = validate_memory_system(test_dir, auto_fix=True)
        
        print(f"   Auto-fix result: {auto_fix_result.get_summary()}")
        print(f"   Issues after auto-fix: {len(auto_fix_result.issues)}")
        
        # Show backup cleanup
        print(f"\n🧹 Backup cleanup (keeping last 30 days)...")
        cleaned_count = error_handler.cleanup_old_backups(days_to_keep=30)
        print(f"   Cleaned up {cleaned_count} old backup files")
        
        # Final validation summary
        print(f"\n📈 FINAL VALIDATION SUMMARY:")
        print("-" * 30)
        
        final_result = validator.validate_memory_system()
        print(f"✅ System validation: {final_result.get_summary()}")
        
        # Show system health
        total_files = len(list(test_dir.rglob("*.md")))
        backup_files = len(list((test_dir / '.backups').glob("*")))
        
        print(f"📊 System Stats:")
        print(f"   Total memory files: {total_files}")
        print(f"   Backup files: {backup_files}")
        print(f"   Validation errors: {final_result.errors_count}")
        print(f"   Validation warnings: {final_result.warnings_count}")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test directory...")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("✅ Cleanup complete")
    
    print(f"\n🎉 Memory validation and error handling demonstration complete!")
    print("="*60)


if __name__ == "__main__":
    demonstrate_validation() 