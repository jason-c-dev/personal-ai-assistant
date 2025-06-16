# Core Memory Directory

This directory contains the most important and persistent memory files that define the user's profile, preferences, and long-term context.

## Core Memory Files

### Essential Files

- **`user_profile.md`** - Complete user profile including personal information, preferences, and background
- **`active_context.md`** - Current context and ongoing projects/conversations
- **`preferences_patterns.md`** - User preferences, communication style, and behavioral patterns
- **`relationship_evolution.md`** - Evolution of the user-assistant relationship over time
- **`life_context.md`** - Important life events, goals, and personal context

### File Characteristics

- **High Importance**: All core files have importance scores of 8-10
- **Persistent**: These files are rarely archived or deleted
- **Frequently Updated**: Content evolves based on ongoing interactions
- **Cross-Referenced**: Files reference and build upon each other

## File Format

Each core memory file follows this structure:

```yaml
---
created: "2025-06-15T10:00:00.000000"
last_updated: "2025-06-15T14:30:22.123456"
importance_score: 10
memory_type: "user_profile"  # or "active_context", "preferences", etc.
category: "core"
version: 3
last_major_update: "2025-06-10T09:15:30.000000"
update_frequency: "high"  # high, medium, low
---

# File Content

Structured markdown content with sections relevant to the memory type...
```

## Memory Types

### User Profile (`user_profile.md`)
- Personal information and background
- Professional details and expertise
- Communication preferences
- Learning style and interaction patterns

### Active Context (`active_context.md`)
- Current projects and goals
- Recent important conversations
- Ongoing tasks and commitments
- Short-term priorities and focus areas

### Preferences & Patterns (`preferences_patterns.md`)
- Communication style preferences
- Behavioral patterns and habits
- Decision-making tendencies
- Preferred tools and methodologies

### Relationship Evolution (`relationship_evolution.md`)
- History of user-assistant interactions
- Trust level and rapport development
- Adaptation of communication style
- Milestone conversations and breakthroughs

### Life Context (`life_context.md`)
- Important life events and changes
- Personal goals and aspirations
- Significant relationships and commitments
- Values and principles

## Privacy and Security

⚠️ **Critical Privacy Notice**: This directory contains highly sensitive personal information.

- **Never commit to version control**: All `.md` files are Git-ignored
- **Local storage only**: Files remain on user's machine
- **Encrypted backups**: Backup system can encrypt core memory files
- **Access control**: Only the memory system should modify these files
- **Audit trail**: All changes are logged for security

## Automated Management

The memory system provides:

### Intelligent Updates
- **Smart Merging**: New information is intelligently integrated
- **Conflict Resolution**: Contradictory information is flagged for review
- **Version Tracking**: Major changes are versioned and tracked
- **Consistency Checks**: Cross-file consistency is maintained

### Protection Mechanisms
- **Backup Priority**: Core files are backed up with highest priority
- **Change Validation**: Significant changes require confirmation
- **Recovery Options**: Multiple recovery points are maintained
- **Integrity Monitoring**: File corruption is detected and reported

### Integration Points
- **Importance Scoring**: Core files maintain high importance scores
- **Search Integration**: Content is indexed for quick retrieval
- **Analytics Tracking**: Usage patterns are monitored
- **Condensation Resistance**: Core files resist automatic summarization

## Usage Guidelines

### For Developers
- Use `CoreMemoryHandlers` class for all file operations
- Never directly modify files - use the memory manager
- Validate changes before applying updates
- Log all modifications for audit purposes

### For Users
- Core memory files are automatically maintained
- Manual editing is not recommended
- Use the assistant interface to update information
- Review periodic summaries of changes

## File Lifecycle

1. **Creation**: Files are created during initial setup or first mention
2. **Evolution**: Content grows and refines through interactions
3. **Maintenance**: Regular updates keep information current
4. **Protection**: Files are protected from accidental deletion
5. **Archival**: Only in extreme cases are core files archived 