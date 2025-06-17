#!/usr/bin/env python3
"""
Migration script to fix file_type to memory_type field inconsistency.
This fixes the 20 validation errors caused by field naming mismatch.
"""

import os
import re
from pathlib import Path

def migrate_file(file_path):
    """Migrate a single memory file from file_type to memory_type."""
    print(f'Migrating: {file_path}')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Handle core files - change file_type: core_memory to specific types
        if 'user_profile.md' in str(file_path):
            content = re.sub(r'file_type:\s*[\'\"]*core_memory[\'\"]*', 'memory_type: user_profile', content)
        elif 'active_context.md' in str(file_path):
            content = re.sub(r'file_type:\s*[\'\"]*core_memory[\'\"]*', 'memory_type: active_context', content)
        elif 'relationship_evolution.md' in str(file_path):
            content = re.sub(r'file_type:\s*[\'\"]*core_memory[\'\"]*', 'memory_type: relationship_evolution', content)
        elif 'preferences_patterns.md' in str(file_path):
            content = re.sub(r'file_type:\s*[\'\"]*core_memory[\'\"]*', 'memory_type: preferences_patterns', content)
        elif 'life_context.md' in str(file_path):
            content = re.sub(r'file_type:\s*[\'\"]*core_memory[\'\"]*', 'memory_type: life_context', content)
        else:
            # Handle interaction files - just change field name
            content = re.sub(r'file_type:\s*([\'\"]*interaction[\'\"]*)', r'memory_type: \1', content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'✅ Successfully migrated: {file_path}')
        else:
            print(f'⏭️  No changes needed: {file_path}')
            
    except Exception as e:
        print(f'❌ Error migrating {file_path}: {e}')

def main():
    """Run the migration."""
    memory_base = Path.home() / '.assistant_memory'
    
    if not memory_base.exists():
        print(f"Memory directory not found: {memory_base}")
        return
    
    print(f"🔄 Starting migration of memory files in {memory_base}")
    
    # Find all .md files
    md_files = list(memory_base.rglob('*.md'))
    print(f"📁 Found {len(md_files)} memory files")
    
    migrated_count = 0
    for md_file in md_files:
        migrate_file(md_file)
        migrated_count += 1
    
    print(f"\n✅ Migration complete! Processed {migrated_count} files")
    print("🧪 Run validation to verify the fix")

if __name__ == "__main__":
    main() 