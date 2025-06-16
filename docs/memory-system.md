# Memory System Guide

Complete guide to understanding and working with the Personal AI Assistant's memory system.

## 🧠 How Memory Works

The assistant creates a personal memory system that mimics how humans remember information - some things are kept in full detail, others are summarized, and the most important facts are preserved long-term.

### Core Principles

- **📱 Human-Readable**: All memories stored as Markdown files you can read and edit
- **🏠 Local-First**: Everything stays on your machine, nothing sent to external servers
- **⚡ Automatic Organization**: Memories sorted by importance and time automatically
- **🤔 Transparent Reasoning**: See why the assistant makes memory decisions
- **✂️ Full Control**: Edit, delete, or organize memories however you want

## 📁 Memory Organization

Your memories live in `~/.assistant_memory/` with this structure:

```
~/.assistant_memory/
├── core/                          # 🎯 Essential facts about you
│   ├── user_profile.md           #     Basic info, role, interests
│   ├── active_context.md         #     Current projects & topics
│   ├── relationship_evolution.md #     How you work with the assistant
│   ├── preferences_patterns.md   #     Communication style & habits
│   └── life_context.md          #     Work, goals, important situations
├── interactions/                  # 📚 Full conversation logs
│   ├── 2024-12/
│   │   ├── 2024-12-15-conversation-001.md
│   │   └── 2024-12-15-conversation-002.md
│   └── 2024-11/
├── condensed/                     # 🗂️ Time-organized summaries
│   ├── recent/     (0-30 days)   #     Full detail, current relevance
│   ├── medium/     (30-180 days) #     Key points, important context
│   └── archive/    (180+ days)   #     Essential facts only
└── system/                        # ⚙️ Assistant operations
    ├── config.json               #     Memory settings
    ├── analytics.db              #     Usage patterns (optional)
    └── backups/                  #     Automatic memory backups
```

## 🎯 Core Memory Files

### User Profile (`core/user_profile.md`)

Contains essential facts about you that rarely change:

```markdown
---
memory_type: core
importance: 10
created: 2024-12-15T10:30:00Z
last_updated: 2024-12-15T14:22:00Z
reasoning: "Core user identity information - highest importance"
---

# User Profile

## Basic Information
- **Name**: Alex Chen
- **Role**: Product Manager at TechStartup Inc.
- **Location**: San Francisco, CA
- **Experience**: 5 years in product management

## Professional Focus
- **Industry**: Mobile fitness applications
- **Current Project**: React Native fitness tracking app
- **Key Challenge**: User retention (improved 40% in Q4 2024)
- **Interests**: UX design, data analytics, user psychology

## Communication Preferences
- **Style**: Direct and practical
- **Detail Level**: Appreciates context and examples
- **Meeting Style**: Prefers structured agendas
- **Follow-up**: Likes action items and next steps
```

### Active Context (`core/active_context.md`)

Current topics and ongoing conversations:

```markdown
---
memory_type: core
importance: 9
created: 2024-12-15T10:30:00Z
last_updated: 2024-12-15T16:45:00Z
reasoning: "Current conversation context - very high importance"
---

# Active Context

## Current Conversations
- **Career Transition**: Exploring move from PM to UX design
- **Fitness App**: User retention improvements and next features
- **Learning**: Taking online UX design course (Coursera)

## Recent Topics
- UX design principles and portfolio building
- A/B testing for mobile onboarding flows
- Transition planning from PM to UX role

## Upcoming Items
- UX portfolio review (mentioned for next week)
- App analytics review (monthly meeting Friday)
- Career planning discussion
```

### Relationship Evolution (`core/relationship_evolution.md`)

How your interaction style with the assistant has developed:

```markdown
---
memory_type: core
importance: 8
created: 2024-12-15T10:30:00Z
last_updated: 2024-12-15T18:20:00Z
reasoning: "Relationship dynamics - important for personalization"
---

# Relationship Evolution

## Interaction History
- **Started**: December 2024
- **Conversation Count**: 47 interactions
- **Primary Use Cases**: Career advice, project planning, problem-solving

## Communication Patterns
- Prefers detailed explanations with examples
- Appreciates when I remember previous context
- Likes actionable advice over theoretical discussions
- Values honest feedback on ideas

## Trust & Comfort Level
- **Initial**: Cautious, testing memory capabilities
- **Current**: Comfortable sharing work challenges and career uncertainties
- **Feedback Style**: Direct and appreciative of honest input
- **Boundaries**: Professional focus, occasional personal context
```

## 📊 Memory Importance System

### Importance Scoring (1-10 Scale)

The assistant rates every memory on importance:

- **10 (Critical)**: Core identity facts, major life events
- **8-9 (High)**: Important preferences, ongoing projects, significant conversations
- **6-7 (Medium)**: Regular interactions, useful context, preferences
- **4-5 (Low-Medium)**: Casual mentions, temporary interests
- **1-3 (Low)**: Small talk, one-off topics, corrected information

### What Influences Importance

1. **Repetition**: Things you mention multiple times
2. **Emotional Context**: Important decisions, challenges, successes
3. **Follow-up Questions**: Topics you return to
4. **Explicit Statements**: "This is important to me"
5. **Professional Relevance**: Work-related context
6. **Personal Goals**: Long-term objectives and aspirations

### Example Importance Ratings

```
10: "I'm considering a career change to UX design" (major life decision)
 9: "Our app's user retention improved 40%" (significant work success)
 8: "I prefer structured meetings with agendas" (important preference)
 7: "I'm taking a UX course on Coursera" (current learning)
 6: "I had a good meeting with the design team" (work context)
 5: "I usually work late on Tuesdays" (routine information)
 3: "I had coffee this morning" (casual mention)
 1: "It's raining today" (irrelevant temporary fact)
```

## 🔄 Memory Lifecycle

### 1. Creation (Real-time)
When you share information, the assistant:
1. **Extracts Key Facts**: Identifies important information
2. **Assigns Importance**: Rates relevance (1-10)
3. **Categorizes**: Determines memory type and location
4. **Stores**: Saves to appropriate memory file
5. **Updates Context**: Modifies active context if relevant

### 2. Organization (Daily)
The system automatically:
1. **Reviews Recent Memories**: Analyzes last 24 hours
2. **Updates Core Files**: Adds new facts to profiles
3. **Resolves Conflicts**: Handles contradictory information
4. **Cross-References**: Links related memories

### 3. Condensation (Weekly/Monthly)
Older memories are intelligently summarized:
1. **Recent → Medium**: After 30 days, detailed memories become summaries
2. **Medium → Archive**: After 180 days, summaries become key facts
3. **Preserves Importance**: High-importance memories retained longer
4. **Maintains Context**: Ensures no loss of crucial information

### 4. Cleanup (Ongoing)
The system maintains health:
1. **Removes Duplicates**: Consolidates redundant information
2. **Corrects Errors**: Updates outdated or incorrect facts
3. **Optimizes Size**: Manages file sizes and storage
4. **Backup Management**: Creates regular backups

## 🔍 How to Use Memory Features

### Viewing Your Memories

```bash
# Browse memory files directly
ls ~/.assistant_memory/core/
cat ~/.assistant_memory/core/user_profile.md

# Use assistant commands during conversation
/memory                    # Show memory overview
/search project           # Search for specific topics
/profile                  # View your user profile
/status                   # Memory system status
```

### Editing Memories

Since memories are Markdown files, you can edit them directly:

```bash
# Edit your profile
code ~/.assistant_memory/core/user_profile.md

# Update preferences
nano ~/.assistant_memory/core/preferences_patterns.md

# Remove sensitive information
rm ~/.assistant_memory/interactions/2024-12/private-conversation.md
```

### Memory Commands During Conversations

- **`What do you remember about my project?`** - Natural memory search
- **`Update my profile: I got promoted`** - Direct memory updates
- **`Forget about that meeting yesterday`** - Memory deletion
- **`Show me what you know about my preferences`** - Memory review

### Advanced Memory Management

```bash
# Backup your memories
cp -r ~/.assistant_memory/ ~/.assistant_memory_backup/

# Reset memory system (keeps conversations)
rm -rf ~/.assistant_memory/core/
python src/main.py  # Will recreate with templates

# Export memories for analysis
grep -r "important_topic" ~/.assistant_memory/ > topic_analysis.txt

# Memory statistics
find ~/.assistant_memory/ -name "*.md" | wc -l  # Count memory files
du -sh ~/.assistant_memory/                     # Memory usage
```

## 🧠 Memory Reasoning System

### Chain-of-Thought Process

The assistant shows its reasoning for memory decisions:

```markdown
## Reasoning Chain

### Information Extraction
- User mentioned "career change to UX design"
- Context: Previous conversations about PM role dissatisfaction
- Emotional indicators: "excited about", "really interested"

### Importance Assessment
- **Base Importance**: 8/10 (career-related major decision)
- **Context Boost**: +1 (aligns with previous concerns)
- **Final Importance**: 9/10

### Storage Decision
- **Location**: core/active_context.md (current ongoing topic)
- **Also Update**: user_profile.md (career interests section)
- **Cross-Reference**: Previous conversations about work satisfaction

### Follow-up Actions
- Monitor for: UX learning progress, timeline decisions
- Remember to ask: How's the UX course going?
- Context for: Career advice and design-related topics
```

### Memory Conflict Resolution

When new information conflicts with existing memories:

```markdown
## Conflict Resolution

### Conflicting Information
- **Previous**: "Uses React Native for main project"  
- **New**: "Decided to switch to Flutter"
- **Context**: 2 weeks between mentions

### Resolution Strategy
- **Action**: Update to new information
- **Reasoning**: Recent decision takes precedence
- **Preservation**: Keep context about the change
- **Result**: "Switched from React Native to Flutter (Dec 2024)"

### Memory Updates
- ✅ Updated project technology in active_context.md
- ✅ Added transition reasoning to user_profile.md
- ✅ Preserved original decision context for learning
```

## 📈 Memory Analytics

### Understanding Your Memory Patterns

The assistant tracks (optionally) how your memory evolves:

```json
{
  "memory_stats": {
    "total_memories": 1247,
    "core_memories": 5,
    "recent_memories": 89,
    "conversation_count": 156,
    "average_importance": 6.3,
    "top_topics": [
      "work_projects",
      "career_development", 
      "technical_interests",
      "communication_preferences"
    ]
  },
  "growth_patterns": {
    "memories_per_week": 12.4,
    "importance_trend": "stable",
    "topic_diversity": "expanding",
    "relationship_depth": "deepening"
  }
}
```

### Memory Health Monitoring

Regular health checks ensure optimal performance:

- **File Integrity**: Validates Markdown and YAML format
- **Size Management**: Monitors storage usage and efficiency  
- **Consistency**: Checks for conflicting information
- **Backup Status**: Ensures memory preservation
- **Performance**: Tracks search and retrieval speed

## 🔧 Customizing Memory Behavior

### Memory Configuration

Customize how memories are handled:

```bash
# .env configuration
MEMORY_BASE_PATH=~/my_ai_memory
MEMORY_RECENT_DAYS=45              # Keep recent memories longer
MEMORY_HIGH_IMPORTANCE_THRESHOLD=8.0   # Stricter importance rating
MEMORY_AUTO_CLEANUP_ENABLED=true      # Automatic maintenance
MEMORY_BACKUP_ENABLED=true            # Regular backups
```

### Custom Memory Templates

Create templates for specific memory types:

```markdown
# custom_project_template.md
---
memory_type: project
importance: 8
template: true
---

# Project: {project_name}

## Overview
- **Start Date**: {start_date}
- **Status**: {status}
- **Technology**: {tech_stack}

## Key Features
{feature_list}

## Challenges & Solutions
{challenges}

## Progress Updates
{progress_log}
```

### Memory Triggers

Set up automatic memory behavior:

```json
{
  "memory_triggers": {
    "project_keywords": ["building", "developing", "working on"],
    "importance_boosters": ["important", "critical", "essential"],
    "relationship_indicators": ["prefer", "like", "dislike", "always"],
    "goal_keywords": ["want to", "planning to", "hope to"]
  }
}
```

## 🚨 Troubleshooting Memory Issues

### Common Memory Problems

#### Memory Not Updating
```bash
# Check memory file permissions
ls -la ~/.assistant_memory/core/

# Verify memory system is enabled
grep MEMORY_ENABLED .env

# Test memory write access
touch ~/.assistant_memory/test_file.md && rm ~/.assistant_memory/test_file.md
```

#### Inconsistent Memories
```bash
# Search for conflicting information
grep -r "conflicting_topic" ~/.assistant_memory/

# Reset specific memory file
cp ~/.assistant_memory/core/user_profile.md ~/backup_profile.md
# Edit file to resolve conflicts
```

#### Performance Issues
```bash
# Check memory directory size
du -sh ~/.assistant_memory/

# Clean up old interactions
find ~/.assistant_memory/interactions/ -name "*.md" -mtime +365 -delete

# Rebuild memory indexes
python -c "
from src.memory.memory_manager import MemoryManager
mm = MemoryManager()
mm.rebuild_indexes()
"
```

### Memory Recovery

If memories are corrupted or lost:

```bash
# Restore from backup
cp -r ~/.assistant_memory_backup/ ~/.assistant_memory/

# Recreate memory system
rm -rf ~/.assistant_memory/
python src/main.py  # Auto-recreates with templates

# Manual memory file repair
# Edit corrupted files - they're just Markdown!
nano ~/.assistant_memory/core/user_profile.md
```

## 🔒 Privacy & Security

### Data Control

- **Local Storage**: All memories stay on your machine
- **No Cloud Sync**: Nothing automatically sent elsewhere  
- **Human Readable**: All files in plain Markdown format
- **Granular Control**: Edit or delete any specific memory
- **Backup Control**: You manage your own backups

### Sensitive Information

```bash
# Remove sensitive memories
rm ~/.assistant_memory/interactions/sensitive-conversation.md

# Edit out sensitive details
nano ~/.assistant_memory/core/user_profile.md

# Exclude patterns from memory
echo "password|credit card|ssn" > ~/.assistant_memory/.memory_ignore
```

### Memory Sharing

When sharing your setup:

```bash
# Safe sharing (removes personal data)
cp -r ~/.assistant_memory/ ~/.assistant_memory_demo/
# Manually edit demo files to remove personal info

# Or start fresh for demos
MEMORY_BASE_PATH=./demo_memory python src/main.py
```

---

*The memory system is designed to enhance your relationship with the AI assistant while keeping you in full control of your personal information.* 🧠✨ 