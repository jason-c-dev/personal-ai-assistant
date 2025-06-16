# Memory Interactions Directory

This directory contains all conversation logs and interaction records between the user and the AI assistant.

## Directory Structure

```
interactions/
├── README.md                    # This file
├── YYYY-MM/                     # Monthly directories
│   ├── README.md               # Monthly summary
│   ├── YYYYMMDD_HHMMSS_interaction.md  # Individual conversations
│   └── daily_summaries/        # Daily interaction summaries
└── recent/                     # Symlinks to most recent interactions
```

## File Naming Convention

- **Individual Interactions**: `YYYYMMDD_HHMMSS_interaction.md`
  - Example: `20250615_143022_interaction.md`
  - Contains full conversation with YAML frontmatter

- **Daily Summaries**: `YYYYMMDD_summary.md`
  - Example: `20250615_summary.md`
  - Condensed summary of the day's interactions

## File Format

Each interaction file contains:

```yaml
---
created: "2025-06-15T14:30:22.123456"
last_updated: "2025-06-15T14:35:45.789012"
importance_score: 7
memory_type: "interaction"
category: "work"
participants: ["user", "assistant"]
topics: ["project planning", "technical discussion"]
summary: "Discussion about project timeline and technical requirements"
---

# Conversation Content

The actual conversation content in markdown format...
```

## Privacy and Data Protection

⚠️ **Important**: This directory contains personal conversation data and should never be committed to version control or shared publicly.

- All `.md` files (except README files) are automatically ignored by Git
- Files are stored locally on the user's machine
- Regular backups are handled by the memory backup system
- Data retention policies are managed by the memory cleanup system

## Automated Management

The memory system automatically:
- Creates monthly directories as needed
- Generates interaction files during conversations
- Updates importance scores based on content analysis
- Creates daily summaries through the condensation system
- Archives old interactions based on retention policies
- Maintains search indexes for quick retrieval

## Integration with Memory System

This directory integrates with:
- **Memory Manager**: File creation and organization
- **Importance Scoring**: Automatic relevance assessment
- **Time-based Organizer**: Monthly/daily structure management
- **Memory Condensation**: Summary generation
- **Memory Search**: Full-text and semantic search
- **Backup System**: Automated data protection
- **Analytics System**: Usage and interaction tracking 