"""
Demo workflow system for Personal AI Assistant.
Provides guided examples that showcase memory capabilities for first-time users.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
from datetime import datetime

console = Console()


class DemoWorkflows:
    """
    Interactive demo system that showcases the assistant's memory capabilities
    through guided workflows and example conversations.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.demo_data_path = self.project_root / "config" / "demo_examples.json"
        self.memory_path = Path(os.getenv('MEMORY_BASE_PATH', '~/.assistant_memory')).expanduser()
    
    def show_welcome_demo(self) -> bool:
        """Show the main demo selection menu."""
        console.print()
        console.print("🎪 [bold cyan]Welcome to your Personal AI Assistant Demo![/bold cyan]")
        console.print()
        
        demo_text = Text()
        demo_text.append("Choose an example workflow to see the assistant in action:\n\n", style="yellow")
        
        workflows = [
            ("1", "👋 Personal Introduction", "See how the assistant learns about you and remembers"),
            ("2", "💼 Project Collaboration", "Work on a project with memory continuity"),
            ("3", "📚 Learning Journey", "Track learning progress over time"),
            ("4", "🛠️ Tech Support", "Get help with technical issues, with context"),
            ("5", "💡 Quick Demo", "See pre-populated memory in action"),
            ("6", "🎮 Interactive Tutorial", "Step-by-step guided experience"),
        ]
        
        for num, title, desc in workflows:
            demo_text.append(f"  {num}. {title}\n", style="bright_blue")
            demo_text.append(f"     {desc}\n\n", style="dim")
        
        demo_text.append("Each demo shows how the assistant:\n", style="yellow")
        demo_text.append("• Remembers what you tell it\n", style="dim")
        demo_text.append("• References past conversations naturally\n", style="dim") 
        demo_text.append("• Builds context over multiple sessions\n", style="dim")
        demo_text.append("• Learns your preferences and work style\n", style="dim")
        
        panel = Panel(demo_text, title="🌟 Demo Workflows", border_style="cyan")
        console.print(panel)
        
        choice = Prompt.ask(
            "\n[bold yellow]Choose a demo[/bold yellow] [dim](1-6, or 'skip' to continue)[/dim]",
            choices=["1", "2", "3", "4", "5", "6", "skip"],
            default="skip"
        )
        
        if choice == "skip":
            return True
        
        # Execute the chosen demo
        demo_map = {
            "1": self.demo_personal_introduction,
            "2": self.demo_project_collaboration, 
            "3": self.demo_learning_journey,
            "4": self.demo_tech_support,
            "5": self.demo_quick_memory,
            "6": self.demo_interactive_tutorial
        }
        
        return demo_map[choice]()
    
    def demo_personal_introduction(self) -> bool:
        """Demo: Personal introduction workflow."""
        console.print("\n🎬 [bold green]Demo: Personal Introduction[/bold green]")
        console.print()
        
        # Show the scenario
        scenario_text = Text()
        scenario_text.append("Scenario: ", style="bold yellow")
        scenario_text.append("You're meeting your AI assistant for the first time. ", style="white")
        scenario_text.append("Watch how it learns about you and stores that information for future conversations.\n\n", style="white")
        
        scenario_text.append("Example conversation:\n", style="cyan bold")
        
        conversation = [
            ("You", "Hi! I'm Alex, a product manager at TechCorp working on mobile apps."),
            ("Assistant", "Nice to meet you, Alex! I'll remember that you're a product manager at TechCorp working on mobile apps. What kind of mobile apps are you developing?"),
            ("You", "We're building a fitness tracking app. I'm particularly focused on user engagement and retention metrics."),
            ("Assistant", "That's fascinating! I've noted that you work on fitness tracking apps with a focus on user engagement and retention. Are there specific challenges you're facing with user retention?"),
            ("You", "Yes, we're seeing a 60% drop-off after the first week. I think it's an onboarding issue."),
            ("Assistant", "I'll remember this retention challenge - 60% drop-off after week one, potentially related to onboarding. This sounds like a crucial problem to solve. Have you done any user research to understand why users are dropping off?")
        ]
        
        for speaker, message in conversation:
            if speaker == "You":
                scenario_text.append(f"\n{speaker}: ", style="bright_blue bold")
                scenario_text.append(f"{message}\n", style="bright_blue")
            else:
                scenario_text.append(f"\n{speaker}: ", style="green bold")
                scenario_text.append(f"{message}\n", style="green")
        
        scenario_text.append("\n💾 Memory Files Created:\n", style="yellow bold")
        scenario_text.append("• user_profile.md - Your role, company, focus areas\n", style="dim")
        scenario_text.append("• active_context.md - Current project and challenges\n", style="dim")
        scenario_text.append("• preferences_patterns.md - Communication style preferences\n", style="dim")
        scenario_text.append("• 2024-12-conversation-001.md - Full conversation log\n", style="dim")
        
        console.print(Panel(scenario_text, title="👋 Personal Introduction Demo", border_style="green"))
        
        # Show the memory impact
        self._show_memory_impact("personal_intro")
        
        return self._demo_continuation_prompt()
    
    def demo_project_collaboration(self) -> bool:
        """Demo: Project collaboration workflow."""
        console.print("\n🎬 [bold green]Demo: Project Collaboration[/bold green]")
        console.print()
        
        scenario_text = Text()
        scenario_text.append("Scenario: ", style="bold yellow")
        scenario_text.append("Working with the assistant on a project over multiple sessions, ", style="white")
        scenario_text.append("showing how context builds and continues across conversations.\n\n", style="white")
        
        scenario_text.append("Session 1 - Project Planning:\n", style="cyan bold")
        conversation_1 = [
            ("You", "I need help planning our Q1 feature release for the fitness app."),
            ("Assistant", "I remember you're working on the fitness tracking app at TechCorp with user retention challenges. What features are you considering for Q1?"),
            ("You", "We want to add social features - friend challenges and leaderboards."),
            ("Assistant", "Great idea! Social features could definitely help with that 60% week-one drop-off issue we discussed. I'll track this Q1 social features project. What's your timeline?")
        ]
        
        for speaker, message in conversation_1:
            if speaker == "You":
                scenario_text.append(f"{speaker}: ", style="bright_blue bold")
                scenario_text.append(f"{message}\n", style="bright_blue")
            else:
                scenario_text.append(f"{speaker}: ", style="green bold")
                scenario_text.append(f"{message}\n", style="green")
        
        scenario_text.append(f"\n[Week later]\n", style="yellow italic")
        scenario_text.append("Session 2 - Progress Check:\n", style="cyan bold")
        
        conversation_2 = [
            ("You", "Quick update on the social features project."),
            ("Assistant", "Perfect! How's the Q1 social features development going? Last time we discussed friend challenges and leaderboards to help with user retention."),
            ("You", "The leaderboards are done, but we're struggling with the friend invitation flow."),
            ("Assistant", "I'll update the project status - leaderboards complete, friend invitations in progress. What specific challenges are you facing with the invitation flow?")
        ]
        
        for speaker, message in conversation_2:
            if speaker == "You":
                scenario_text.append(f"{speaker}: ", style="bright_blue bold") 
                scenario_text.append(f"{message}\n", style="bright_blue")
            else:
                scenario_text.append(f"{speaker}: ", style="green bold")
                scenario_text.append(f"{message}\n", style="green")
        
        scenario_text.append("\n🧠 Memory Continuity:\n", style="yellow bold")
        scenario_text.append("• Remembers your role and company context\n", style="dim")
        scenario_text.append("• Connects new project to existing retention challenge\n", style="dim")
        scenario_text.append("• Tracks project progress across sessions\n", style="dim")
        scenario_text.append("• Maintains conversation context week-to-week\n", style="dim")
        
        console.print(Panel(scenario_text, title="💼 Project Collaboration Demo", border_style="blue"))
        
        self._show_memory_impact("project_collab")
        
        return self._demo_continuation_prompt()
    
    def demo_learning_journey(self) -> bool:
        """Demo: Learning journey workflow."""
        console.print("\n🎬 [bold green]Demo: Learning Journey[/bold green]")
        console.print()
        
        scenario_text = Text()
        scenario_text.append("Scenario: ", style="bold yellow")
        scenario_text.append("Using the assistant as a learning companion that tracks your progress ", style="white")
        scenario_text.append("and adapts to your learning style over time.\n\n", style="white")
        
        scenario_text.append("Learning Session 1:\n", style="cyan bold")
        conversation_1 = [
            ("You", "I want to learn React Native to help with our mobile app development."),
            ("Assistant", "That makes perfect sense for your role as a product manager at TechCorp working on the fitness app! Learning React Native will help you better collaborate with your dev team. What's your current experience with React or mobile development?"),
            ("You", "I know basic JavaScript but no React. I prefer hands-on learning over reading docs."),
            ("Assistant", "Got it! I'll remember you prefer hands-on learning. Let me suggest starting with a simple React Native tutorial project rather than diving into documentation. Would you like me to recommend some practical tutorials?")
        ]
        
        for speaker, message in conversation_1:
            if speaker == "You":
                scenario_text.append(f"{speaker}: ", style="bright_blue bold")
                scenario_text.append(f"{message}\n", style="bright_blue")
            else:
                scenario_text.append(f"{speaker}: ", style="green bold")
                scenario_text.append(f"{message}\n", style="green")
        
        scenario_text.append(f"\n[Two weeks later]\n", style="yellow italic")
        scenario_text.append("Progress Check:\n", style="cyan bold")
        
        conversation_2 = [
            ("You", "Update on my React Native learning - I built a simple todo app!"),
            ("Assistant", "That's fantastic progress! I remember you prefer hands-on learning, so building a todo app was perfect. How did you find the development experience? Any concepts that clicked or were challenging?"),
            ("You", "State management was confusing at first, but components make sense now."),
            ("Assistant", "Great insight! Since state management was tricky and you learn best by doing, should we look at a practical state management example using your fitness app context?")
        ]
        
        for speaker, message in conversation_2:
            if speaker == "You":
                scenario_text.append(f"{speaker}: ", style="bright_blue bold")
                scenario_text.append(f"{message}\n", style="bright_blue")
            else:
                scenario_text.append(f"{speaker}: ", style="green bold")
                scenario_text.append(f"{message}\n", style="green")
        
        scenario_text.append("\n📈 Learning Adaptation:\n", style="yellow bold")
        scenario_text.append("• Remembers your learning style (hands-on vs docs)\n", style="dim")
        scenario_text.append("• Connects learning to your work context\n", style="dim")
        scenario_text.append("• Tracks progress and builds on previous sessions\n", style="dim")
        scenario_text.append("• Adapts suggestions based on what worked/didn't work\n", style="dim")
        
        console.print(Panel(scenario_text, title="📚 Learning Journey Demo", border_style="purple"))
        
        self._show_memory_impact("learning_journey")
        
        return self._demo_continuation_prompt()
    
    def demo_tech_support(self) -> bool:
        """Demo: Technical support with context."""
        console.print("\n🎬 [bold green]Demo: Technical Support[/bold green]")
        console.print()
        
        scenario_text = Text()
        scenario_text.append("Scenario: ", style="bold yellow")
        scenario_text.append("Getting technical help where the assistant remembers your setup, ", style="white")
        scenario_text.append("previous issues, and provides contextual solutions.\n\n", style="white")
        
        scenario_text.append("Support Session:\n", style="cyan bold")
        conversation = [
            ("You", "I'm having trouble with our React Native build failing on iOS."),
            ("Assistant", "I remember you're working on the fitness tracking app at TechCorp and recently learning React Native. Let me help debug this iOS build issue. What error message are you seeing?"),
            ("You", "Something about missing provisioning profiles."),
            ("Assistant", "Ah, that's a common iOS setup issue. Since you mentioned you prefer hands-on learning over docs, let me walk you through the practical steps to fix this rather than sending you to Apple's documentation. First, let's check your Xcode settings..."),
            ("You", "Actually, can you just give me the quick fix? I need to get this build working for a demo."),
            ("Assistant", "Absolutely! I can see you need a quick solution right now. Here's the fastest fix: [provides specific commands]. I'll also remember that sometimes you need quick fixes vs. detailed explanations, depending on the situation.")
        ]
        
        for speaker, message in conversation:
            if speaker == "You":
                scenario_text.append(f"{speaker}: ", style="bright_blue bold")
                scenario_text.append(f"{message}\n", style="bright_blue")
            else:
                scenario_text.append(f"{speaker}: ", style="green bold")
                scenario_text.append(f"{message}\n", style="green")
        
        scenario_text.append("\n🔧 Contextual Support:\n", style="yellow bold")
        scenario_text.append("• Knows your tech stack and current projects\n", style="dim")
        scenario_text.append("• Remembers your learning preferences\n", style="dim")
        scenario_text.append("• Adapts explanation style to current needs\n", style="dim")
        scenario_text.append("• Builds knowledge of your common issues\n", style="dim")
        
        console.print(Panel(scenario_text, title="🛠️ Technical Support Demo", border_style="red"))
        
        self._show_memory_impact("tech_support")
        
        return self._demo_continuation_prompt()
    
    def demo_quick_memory(self) -> bool:
        """Demo: Quick memory showcase with pre-populated data."""
        console.print("\n🎬 [bold green]Demo: Memory in Action[/bold green]")
        console.print()
        
        # Create sample memory files
        self._create_sample_memory_files()
        
        scenario_text = Text()
        scenario_text.append("🧠 I've created some sample memory files to show you how the assistant remembers information:\n\n", style="yellow")
        
        # Show memory files created
        memory_files = [
            ("user_profile.md", "Your role: Senior Developer at InnovateCorp"),
            ("active_context.md", "Current project: E-commerce platform redesign"),
            ("preferences_patterns.md", "Prefers: Code examples over theory"),
            ("relationship_evolution.md", "Working relationship: Direct, technical discussions")
        ]
        
        for filename, content in memory_files:
            scenario_text.append(f"📄 {filename}\n", style="cyan bold")
            scenario_text.append(f"   {content}\n\n", style="dim")
        
        scenario_text.append("Now try asking the assistant:\n", style="yellow bold")
        examples = [
            "\"What am I working on right now?\"",
            "\"What's my preferred learning style?\"",
            "\"Tell me about our working relationship\"",
            "\"What do you remember about me?\""
        ]
        
        for example in examples:
            scenario_text.append(f"• {example}\n", style="bright_blue")
        
        scenario_text.append(f"\nThe assistant will reference these memory files naturally in its responses!\n", style="green")
        
        console.print(Panel(scenario_text, title="💡 Quick Memory Demo", border_style="yellow"))
        
        console.print(f"\n📁 [cyan]Memory files created at:[/cyan] {self.memory_path}")
        console.print("   [dim]You can view and edit these files directly![/dim]")
        
        return self._demo_continuation_prompt()
    
    def demo_interactive_tutorial(self) -> bool:
        """Demo: Interactive step-by-step tutorial."""
        console.print("\n🎬 [bold green]Demo: Interactive Tutorial[/bold green]")
        console.print()
        
        tutorial_text = Text()
        tutorial_text.append("🎓 Let's walk through the assistant's memory system step by step:\n\n", style="yellow bold")
        
        steps = [
            ("Step 1: Introduction", "Tell the assistant about yourself and watch it create your profile"),
            ("Step 2: Set Context", "Share your current projects and see active context tracking"),
            ("Step 3: Express Preferences", "Mention how you like to work and communicate"),
            ("Step 4: Continue Conversation", "See how it references everything you've shared"),
            ("Step 5: End & Return", "Come back later and watch it remember everything")
        ]
        
        for step_title, step_desc in steps:
            tutorial_text.append(f"📚 {step_title}\n", style="cyan bold")
            tutorial_text.append(f"   {step_desc}\n\n", style="white")
        
        tutorial_text.append("💫 Key Features You'll Experience:\n", style="yellow bold")
        features = [
            "Personal profile building and updates",
            "Project and context tracking",
            "Learning style and preference recognition", 
            "Natural conversation continuity",
            "Cross-session memory persistence"
        ]
        
        for feature in features:
            tutorial_text.append(f"• {feature}\n", style="dim")
        
        tutorial_text.append(f"\n🚀 Ready to start? Just begin chatting with the assistant!\n", style="green bold")
        
        console.print(Panel(tutorial_text, title="🎮 Interactive Tutorial", border_style="magenta"))
        
        return self._demo_continuation_prompt()
    
    def _show_memory_impact(self, demo_type: str):
        """Show what memory files would be created/updated."""
        console.print(f"\n📋 [yellow]Memory Impact:[/yellow]")
        
        impact_map = {
            "personal_intro": [
                ("user_profile.md", "Created", "Basic info: role, company, focus areas"),
                ("active_context.md", "Created", "Current challenges and projects"),
                ("preferences_patterns.md", "Created", "Communication preferences noted"),
                ("interactions/2024-12/", "Created", "Full conversation logged")
            ],
            "project_collab": [
                ("active_context.md", "Updated", "Q1 social features project added"),
                ("interactions/", "Updated", "Multiple session conversations"),
                ("condensed/recent/", "Created", "Project progress summaries"),
                ("relationship_evolution.md", "Updated", "Collaboration patterns learned")
            ],
            "learning_journey": [
                ("active_context.md", "Updated", "React Native learning goal added"),
                ("preferences_patterns.md", "Updated", "Hands-on learning preference"),
                ("condensed/recent/", "Updated", "Learning progress tracked"),
                ("interactions/", "Updated", "Learning sessions logged")
            ],
            "tech_support": [
                ("user_profile.md", "Updated", "Tech stack and tools noted"),
                ("preferences_patterns.md", "Updated", "Support style preferences"),
                ("interactions/", "Updated", "Technical issues and solutions"),
                ("condensed/recent/", "Updated", "Common problems patterns")
            ]
        }
        
        if demo_type in impact_map:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("File", style="bright_blue")
            table.add_column("Action", style="yellow")
            table.add_column("Content", style="white")
            
            for filename, action, content in impact_map[demo_type]:
                table.add_row(filename, action, content)
            
            console.print(table)
    
    def _create_sample_memory_files(self):
        """Create sample memory files for the quick demo."""
        # Ensure memory directory exists
        self.memory_path.mkdir(parents=True, exist_ok=True)
        (self.memory_path / "core").mkdir(exist_ok=True)
        
        sample_files = {
            "core/user_profile.md": """---
created: 2024-12-16
updated: 2024-12-16
importance: 10
tags: [profile, work, role]
---

# User Profile

## Basic Information
- **Name**: Sample User
- **Role**: Senior Developer
- **Company**: InnovateCorp
- **Location**: Remote

## Professional Focus
- Full-stack development
- E-commerce platforms
- Performance optimization
- Team leadership

## Current Responsibilities
- Leading the platform redesign project
- Mentoring junior developers
- Architecture decisions
""",
            "core/active_context.md": """---
created: 2024-12-16
updated: 2024-12-16
importance: 9
tags: [current, project, priorities]
---

# Active Context

## Current Projects
### E-commerce Platform Redesign (Primary)
- **Status**: In progress (Month 2 of 4)
- **Goal**: Improve performance and user experience
- **Challenges**: Legacy code integration, performance bottlenecks
- **Timeline**: Q1 completion target

## Immediate Priorities
1. Database optimization review
2. Frontend component migration
3. API performance testing
4. Team sync meetings

## Current Learning
- GraphQL implementation
- Microservices patterns
- React performance optimization
""",
            "core/preferences_patterns.md": """---
created: 2024-12-16 
updated: 2024-12-16
importance: 8
tags: [preferences, communication, style]
---

# Preferences & Patterns

## Communication Style
- Prefers direct, technical discussions
- Likes code examples over abstract explanations
- Values practical solutions
- Appreciates concise responses

## Learning Preferences
- Hands-on experimentation
- Code examples and demos
- Progressive complexity
- Real-world applications

## Work Style
- Morning person (most productive 9-11 AM)
- Prefers focused work blocks
- Regular breaks for optimization
- Collaborative decision making
""",
            "core/relationship_evolution.md": """---
created: 2024-12-16
updated: 2024-12-16
importance: 7
tags: [relationship, interaction, growth]
---

# Relationship Evolution

## Interaction History
- Started with technical questions
- Evolved to project collaboration
- Built trust through practical advice
- Established ongoing partnership

## Working Dynamic
- Technical advisor and sounding board
- Project planning assistant
- Learning companion for new technologies
- Problem-solving partner

## Strengths Recognized
- Strong technical foundation
- Good system thinking
- Team leadership skills
- Performance optimization expertise

## Areas of Growth
- Exploring new frameworks
- Scaling team processes
- Architecture decision confidence
"""
        }
        
        for filepath, content in sample_files.items():
            full_path = self.memory_path / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content.strip())
    
    def _demo_continuation_prompt(self) -> bool:
        """Ask user if they want to continue with more demos or start using the assistant."""
        console.print()
        
        choice = Prompt.ask(
            "[bold yellow]What would you like to do next?[/bold yellow]",
            choices=["more", "start", "menu"],
            default="start"
        )
        
        if choice == "more":
            return self.show_welcome_demo()
        elif choice == "menu":
            return self.show_welcome_demo()
        else:
            console.print("\n🎉 [bold green]Ready to start using your AI assistant![/bold green]")
            console.print("   [dim]Just begin chatting and watch the memory system work its magic![/dim]")
            return True


def show_demo_workflows() -> bool:
    """
    Main entry point for demo workflows.
    Returns True if user wants to continue to main app.
    """
    demo = DemoWorkflows()
    return demo.show_welcome_demo()


if __name__ == "__main__":
    show_demo_workflows() 