"""
Configuration utilities for the Personal AI Assistant.

This module handles loading and accessing configuration settings from 
environment variables and configuration files.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def get_memory_base_path() -> str:
    """
    Get the base path for memory storage.
    
    Returns:
        str: Path to memory storage directory
    """
    default_path = "~/assistant_memory"
    return os.getenv('MEMORY_BASE_PATH', default_path)


def get_memory_recent_days() -> int:
    """
    Get the number of days for recent memory retention.
    
    Returns:
        int: Number of days for recent memories
    """
    return int(os.getenv('MEMORY_RECENT_DAYS', '30'))


def get_memory_medium_days() -> int:
    """
    Get the number of days for medium-term memory retention.
    
    Returns:
        int: Number of days for medium-term memories
    """
    return int(os.getenv('MEMORY_MEDIUM_DAYS', '180'))


def get_memory_archive_days() -> int:
    """
    Get the number of days for archive memory retention.
    
    Returns:
        int: Number of days for archive memories
    """
    return int(os.getenv('MEMORY_ARCHIVE_DAYS', '180'))


def get_memory_max_file_size_mb() -> int:
    """
    Get the maximum file size for memory files in MB.
    
    Returns:
        int: Maximum file size in MB
    """
    return int(os.getenv('MEMORY_MAX_FILE_SIZE_MB', '5'))


def get_memory_high_importance_threshold() -> int:
    """
    Get the threshold for high importance memories.
    
    Returns:
        int: High importance threshold (1-10 scale)
    """
    return int(os.getenv('MEMORY_HIGH_IMPORTANCE_THRESHOLD', '7'))


def get_memory_medium_importance_threshold() -> int:
    """
    Get the threshold for medium importance memories.
    
    Returns:
        int: Medium importance threshold (1-10 scale)
    """
    return int(os.getenv('MEMORY_MEDIUM_IMPORTANCE_THRESHOLD', '4'))


def is_memory_backup_enabled() -> bool:
    """
    Check if memory backup is enabled.
    
    Returns:
        bool: True if backup is enabled
    """
    return os.getenv('MEMORY_BACKUP_ENABLED', 'true').lower() == 'true'


def get_ai_provider() -> str:
    """
    Get the AI provider setting.
    
    Returns:
        str: AI provider name (anthropic, openai, bedrock)
    """
    return os.getenv('AI_PROVIDER', 'anthropic')


def get_anthropic_api_key() -> Optional[str]:
    """
    Get the Anthropic API key.
    
    Returns:
        Optional[str]: Anthropic API key if set
    """
    return os.getenv('ANTHROPIC_API_KEY')


def get_openai_api_key() -> Optional[str]:
    """
    Get the OpenAI API key.
    
    Returns:
        Optional[str]: OpenAI API key if set
    """
    return os.getenv('OPENAI_API_KEY')


def get_aws_credentials() -> Dict[str, Optional[str]]:
    """
    Get AWS credentials for Bedrock.
    
    Returns:
        Dict containing AWS credentials
    """
    return {
        'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'region': os.getenv('AWS_REGION', 'us-east-1')
    }


def get_system_prompt_file() -> str:
    """
    Get the path to the system prompt configuration file.
    
    Returns:
        str: Path to system prompt file
    """
    return os.getenv('SYSTEM_PROMPT_FILE', 'config/system_prompts.json')


def get_model_config_file() -> str:
    """
    Get the path to the model configuration file.
    
    Returns:
        str: Path to model config file
    """
    return os.getenv('MODEL_CONFIG_FILE', 'config/model_config.json')


def get_log_level() -> str:
    """
    Get the logging level.
    
    Returns:
        str: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    return os.getenv('LOG_LEVEL', 'INFO')


def is_debug_mode() -> bool:
    """
    Check if debug mode is enabled.
    
    Returns:
        bool: True if debug mode is enabled
    """
    return os.getenv('DEBUG', 'false').lower() == 'true'


def get_max_context_tokens() -> int:
    """
    Get the maximum context tokens for AI requests.
    
    Returns:
        int: Maximum context tokens
    """
    return int(os.getenv('MAX_CONTEXT_TOKENS', '100000'))


def get_max_response_tokens() -> int:
    """
    Get the maximum response tokens for AI requests.
    
    Returns:
        int: Maximum response tokens
    """
    return int(os.getenv('MAX_RESPONSE_TOKENS', '4000'))


def get_temperature() -> float:
    """
    Get the temperature setting for AI responses.
    
    Returns:
        float: Temperature value (0.0-1.0)
    """
    return float(os.getenv('TEMPERATURE', '0.7'))


def get_session_timeout_minutes() -> int:
    """
    Get the session timeout in minutes.
    
    Returns:
        int: Session timeout in minutes
    """
    return int(os.getenv('SESSION_TIMEOUT_MINUTES', '60'))


def get_auto_save_interval_seconds() -> int:
    """
    Get the auto-save interval in seconds.
    
    Returns:
        int: Auto-save interval in seconds
    """
    return int(os.getenv('AUTO_SAVE_INTERVAL_SECONDS', '30'))


def validate_environment() -> Dict[str, Any]:
    """
    Validate the environment configuration.
    
    Returns:
        Dict containing validation results
    """
    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check AI provider configuration
    provider = get_ai_provider()
    
    if provider == 'anthropic':
        if not get_anthropic_api_key():
            validation['errors'].append("ANTHROPIC_API_KEY is required when using Anthropic provider")
            validation['valid'] = False
    elif provider == 'openai':
        if not get_openai_api_key():
            validation['errors'].append("OPENAI_API_KEY is required when using OpenAI provider")
            validation['valid'] = False
    elif provider == 'bedrock':
        aws_creds = get_aws_credentials()
        if not aws_creds['access_key_id'] or not aws_creds['secret_access_key']:
            validation['errors'].append("AWS credentials are required when using Bedrock provider")
            validation['valid'] = False
    else:
        validation['errors'].append(f"Unknown AI provider: {provider}")
        validation['valid'] = False
    
    # Check memory configuration
    memory_path = Path(get_memory_base_path()).expanduser()
    if not memory_path.parent.exists():
        validation['warnings'].append(f"Memory base path parent directory does not exist: {memory_path.parent}")
    
    # Check configuration files
    prompt_file = Path(get_system_prompt_file())
    if not prompt_file.exists():
        validation['warnings'].append(f"System prompt file not found: {prompt_file}")
    
    model_file = Path(get_model_config_file())
    if not model_file.exists():
        validation['warnings'].append(f"Model config file not found: {model_file}")
    
    return validation


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of current configuration.
    
    Returns:
        Dict containing configuration summary
    """
    return {
        'ai_provider': get_ai_provider(),
        'memory_base_path': get_memory_base_path(),
        'memory_retention': {
            'recent_days': get_memory_recent_days(),
            'medium_days': get_memory_medium_days(),
            'archive_days': get_memory_archive_days()
        },
        'memory_limits': {
            'max_file_size_mb': get_memory_max_file_size_mb(),
            'backup_enabled': is_memory_backup_enabled()
        },
        'importance_thresholds': {
            'high': get_memory_high_importance_threshold(),
            'medium': get_memory_medium_importance_threshold()
        },
        'ai_settings': {
            'max_context_tokens': get_max_context_tokens(),
            'max_response_tokens': get_max_response_tokens(),
            'temperature': get_temperature()
        },
        'session_settings': {
            'timeout_minutes': get_session_timeout_minutes(),
            'auto_save_interval_seconds': get_auto_save_interval_seconds()
        },
        'debug_mode': is_debug_mode(),
        'log_level': get_log_level()
    } 