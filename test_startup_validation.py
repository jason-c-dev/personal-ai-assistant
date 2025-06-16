#!/usr/bin/env python3
"""
Test script for the enhanced startup validation system.
Demonstrates the new error messages and health dashboard.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_startup_validation():
    """Test the enhanced startup validation system."""
    from src.utils.startup_validator import validate_startup
    
    print("🧪 Testing Enhanced Startup Validation System")
    print("=" * 60)
    
    # Run the validation
    success = validate_startup()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Validation passed! System is ready for use.")
    else:
        print("❌ Validation failed. Please follow the fix suggestions above.")
    
    return success

def test_individual_checks():
    """Test individual validation components."""
    from src.utils.startup_validator import StartupValidator
    
    print("\n🔬 Testing Individual Validation Components")
    print("=" * 60)
    
    validator = StartupValidator()
    
    # Test each component individually
    checks = [
        ("Python Version", validator._validate_python_version),
        ("Environment File", validator._validate_environment_file),
        ("API Key", validator._validate_api_key),
        ("Dependencies", validator._validate_dependencies),
        ("Config Files", validator._validate_config_files),
        ("Memory System", validator._validate_memory_system),
        ("Network Access", validator._validate_network_connectivity),
        ("File Permissions", validator._validate_file_permissions),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result
            status = "✅" if result['status'] else "❌"
            print(f"{status} {name}: {result['message']}")
            if not result['status'] and 'fix' in result:
                print(f"   Fix: {result['fix']}")
        except Exception as e:
            print(f"❌ {name}: Error during check - {e}")
            results[name] = {'status': False, 'message': str(e)}
    
    return results

if __name__ == "__main__":
    print("🚀 Personal AI Assistant - Startup Validation Test")
    print("=" * 60)
    
    # Test the main validation function
    main_success = test_startup_validation()
    
    # Test individual components for debugging
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        individual_results = test_individual_checks()
        
        print(f"\n📊 Summary:")
        print(f"   Main validation: {'✅ PASSED' if main_success else '❌ FAILED'}")
        
        passed = sum(1 for r in individual_results.values() if r['status'])
        total = len(individual_results)
        print(f"   Individual checks: {passed}/{total} passed")
        
        if passed < total:
            print(f"\n⚠️  Failed checks:")
            for name, result in individual_results.items():
                if not result['status']:
                    print(f"   - {name}: {result['message']}")
    
    sys.exit(0 if main_success else 1) 