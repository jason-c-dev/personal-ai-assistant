"""
Session Management System

Handles conversation session persistence, state management, and restoration
for the Personal AI Assistant CLI.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import uuid

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


class SessionState:
    """Represents the complete state of a conversation session."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.title = ""
        self.description = ""
        self.conversation_turns: List[Dict[str, Any]] = []
        self.context_summary = ""
        self.user_metadata: Dict[str, Any] = {}
        self.agent_metadata: Dict[str, Any] = {}
        self.total_messages = 0
        self.session_duration_seconds = 0
        self.tags: List[str] = []
        self.bookmarked = False
        self.auto_save = True
        
    def add_conversation_turn(self, user_message: str, assistant_response: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a conversation turn to the session."""
        turn = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'assistant_response': assistant_response,
            'turn_number': len(self.conversation_turns) + 1,
            'metadata': metadata or {}
        }
        self.conversation_turns.append(turn)
        self.total_messages += 1
        self.last_accessed = datetime.now()
        
        # Auto-generate title from first meaningful exchange
        if not self.title and len(self.conversation_turns) == 1:
            self.title = self._generate_title_from_first_turn(user_message)
    
    def _generate_title_from_first_turn(self, user_message: str) -> str:
        """Generate a session title from the first user message."""
        # Clean and truncate the message for title
        title = user_message.strip()
        
        # Remove common greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        words = title.lower().split()
        if words and words[0] in greetings:
            title = ' '.join(words[1:]) if len(words) > 1 else title
        
        # Truncate to reasonable length
        if len(title) > 50:
            title = title[:47] + "..."
        
        return title or f"Session {self.session_id[:8]}"
    
    def update_context_summary(self, summary: str):
        """Update the session's context summary."""
        self.context_summary = summary
        self.last_accessed = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag to the session."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.last_accessed = datetime.now()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the session."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.last_accessed = datetime.now()
    
    def update_duration(self, start_time: datetime):
        """Update session duration based on start time."""
        self.session_duration_seconds = (datetime.now() - start_time).total_seconds()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the session."""
        return {
            'session_id': self.session_id,
            'title': self.title,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'total_turns': len(self.conversation_turns),
            'total_messages': self.total_messages,
            'duration_minutes': round(self.session_duration_seconds / 60, 1),
            'tags': self.tags,
            'bookmarked': self.bookmarked,
            'has_context': bool(self.context_summary)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'title': self.title,
            'description': self.description,
            'conversation_turns': self.conversation_turns,
            'context_summary': self.context_summary,
            'user_metadata': self.user_metadata,
            'agent_metadata': self.agent_metadata,
            'total_messages': self.total_messages,
            'session_duration_seconds': self.session_duration_seconds,
            'tags': self.tags,
            'bookmarked': self.bookmarked,
            'auto_save': self.auto_save
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create session state from dictionary."""
        session = cls(session_id=data.get('session_id'))
        session.created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        session.last_accessed = datetime.fromisoformat(data.get('last_accessed', datetime.now().isoformat()))
        session.title = data.get('title', '')
        session.description = data.get('description', '')
        session.conversation_turns = data.get('conversation_turns', [])
        session.context_summary = data.get('context_summary', '')
        session.user_metadata = data.get('user_metadata', {})
        session.agent_metadata = data.get('agent_metadata', {})
        session.total_messages = data.get('total_messages', 0)
        session.session_duration_seconds = data.get('session_duration_seconds', 0)
        session.tags = data.get('tags', [])
        session.bookmarked = data.get('bookmarked', False)
        session.auto_save = data.get('auto_save', True)
        return session


class SessionManager:
    """Manages conversation session persistence and restoration."""
    
    def __init__(self, sessions_dir: Optional[Path] = None):
        self.sessions_dir = sessions_dir or Path("memory/sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Session cache for performance
        self._session_cache: Dict[str, SessionState] = {}
        self._cache_loaded = False
        
        # Current active session
        self.current_session: Optional[SessionState] = None
        
        # Session metadata index
        self.session_index_file = self.sessions_dir / "session_index.json"
        self._session_index: Dict[str, Dict[str, Any]] = {}
        
        # Auto-save settings
        self.auto_save_enabled = True
        self.auto_save_interval = 30  # seconds
        self._last_auto_save = datetime.now()
        
    async def initialize(self):
        """Initialize the session manager and load existing sessions."""
        try:
            await self._load_session_index()
            console.print("[dim]📂 Session manager initialized[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not load session index: {e}[/yellow]")
            self._session_index = {}
    
    async def _load_session_index(self):
        """Load the session index from disk."""
        if self.session_index_file.exists():
            with open(self.session_index_file, 'r') as f:
                self._session_index = json.load(f)
        else:
            self._session_index = {}
    
    async def _save_session_index(self):
        """Save the session index to disk."""
        try:
            with open(self.session_index_file, 'w') as f:
                json.dump(self._session_index, f, indent=2)
        except Exception as e:
            console.print(f"[red]❌ Failed to save session index: {e}[/red]")
    
    def create_new_session(self, title: Optional[str] = None) -> SessionState:
        """Create a new conversation session."""
        session = SessionState()
        if title:
            session.title = title
        
        self.current_session = session
        self._session_cache[session.session_id] = session
        
        # Update index
        self._session_index[session.session_id] = session.get_summary()
        
        console.print(f"[green]✨ New session created: {session.session_id[:8]}[/green]")
        return session
    
    async def save_session(self, session: Optional[SessionState] = None, force: bool = False) -> bool:
        """Save a session to disk."""
        if session is None:
            session = self.current_session
        
        if not session:
            return False
        
        if not force and not session.auto_save:
            return False
        
        try:
            session_file = self.sessions_dir / f"{session.session_id}.json"
            
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            
            # Update index
            self._session_index[session.session_id] = session.get_summary()
            await self._save_session_index()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save session {session.session_id[:8]}: {e}[/red]")
            return False
    
    async def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load a session from disk."""
        # Check cache first
        if session_id in self._session_cache:
            return self._session_cache[session_id]
        
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            
            if not session_file.exists():
                console.print(f"[red]❌ Session {session_id[:8]} not found[/red]")
                return None
            
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            session = SessionState.from_dict(data)
            self._session_cache[session_id] = session
            
            return session
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load session {session_id[:8]}: {e}[/red]")
            return None
    
    async def restore_session(self, session_id: str) -> bool:
        """Restore a session as the current active session."""
        session = await self.load_session(session_id)
        
        if session:
            self.current_session = session
            session.last_accessed = datetime.now()
            
            # Auto-save the access update
            await self.save_session(session)
            
            console.print(f"[green]🔄 Session restored: {session.title or session.session_id[:8]}[/green]")
            return True
        
        return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from disk and cache."""
        try:
            # Remove from cache
            if session_id in self._session_cache:
                del self._session_cache[session_id]
            
            # Remove from index
            if session_id in self._session_index:
                del self._session_index[session_id]
            
            # Remove file
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Save updated index
            await self._save_session_index()
            
            console.print(f"[green]🗑️  Session deleted: {session_id[:8]}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to delete session {session_id[:8]}: {e}[/red]")
            return False
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions sorted by last accessed time."""
        sessions = list(self._session_index.values())
        sessions.sort(key=lambda x: x['last_accessed'], reverse=True)
        return sessions[:limit]
    
    def search_sessions(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search sessions by title, description, or content."""
        query_lower = query.lower()
        matches = []
        
        for session_summary in self._session_index.values():
            score = 0
            
            # Check title
            if query_lower in session_summary.get('title', '').lower():
                score += 10
            
            # Check description
            if query_lower in session_summary.get('description', '').lower():
                score += 5
            
            # Check tags
            for tag in session_summary.get('tags', []):
                if query_lower in tag.lower():
                    score += 3
            
            if score > 0:
                matches.append((score, session_summary))
        
        # Sort by score and return
        matches.sort(key=lambda x: x[0], reverse=True)
        return [match[1] for match in matches[:limit]]
    
    def get_sessions_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Get sessions that contain any of the specified tags."""
        matches = []
        
        for session_summary in self._session_index.values():
            session_tags = set(session_summary.get('tags', []))
            if session_tags.intersection(set(tags)):
                matches.append(session_summary)
        
        return matches
    
    def get_bookmarked_sessions(self) -> List[Dict[str, Any]]:
        """Get all bookmarked sessions."""
        return [s for s in self._session_index.values() if s.get('bookmarked', False)]
    
    async def auto_save_check(self):
        """Check if auto-save should be performed."""
        if (self.auto_save_enabled and 
            self.current_session and 
            self.current_session.auto_save and
            (datetime.now() - self._last_auto_save).total_seconds() >= self.auto_save_interval):
            
            await self.save_session(self.current_session)
            self._last_auto_save = datetime.now()
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about all sessions."""
        total_sessions = len(self._session_index)
        
        if total_sessions == 0:
            return {
                'total_sessions': 0,
                'total_conversations': 0,
                'total_messages': 0,
                'avg_session_length': 0,
                'bookmarked_sessions': 0,
                'sessions_with_tags': 0
            }
        
        total_conversations = sum(s.get('total_turns', 0) for s in self._session_index.values())
        total_messages = sum(s.get('total_messages', 0) for s in self._session_index.values())
        avg_session_length = total_conversations / total_sessions if total_sessions > 0 else 0
        bookmarked_sessions = sum(1 for s in self._session_index.values() if s.get('bookmarked', False))
        sessions_with_tags = sum(1 for s in self._session_index.values() if s.get('tags', []))
        
        return {
            'total_sessions': total_sessions,
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'avg_session_length': round(avg_session_length, 1),
            'bookmarked_sessions': bookmarked_sessions,
            'sessions_with_tags': sessions_with_tags
        }
    
    async def cleanup_old_sessions(self, days_to_keep: int = 30) -> int:
        """Clean up sessions older than specified days (excluding bookmarked)."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        sessions_to_delete = []
        
        for session_id, session_summary in self._session_index.items():
            if session_summary.get('bookmarked', False):
                continue  # Skip bookmarked sessions
            
            last_accessed = datetime.fromisoformat(session_summary['last_accessed'])
            if last_accessed < cutoff_date:
                sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            if await self.delete_session(session_id):
                deleted_count += 1
        
        return deleted_count 