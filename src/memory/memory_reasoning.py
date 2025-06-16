"""
Memory Decision Reasoning Engine

This module provides comprehensive chain-of-thought reasoning for all memory decisions
including storage, retrieval, modification, condensation, and prioritization operations.
It creates detailed reasoning trails, confidence assessments, and alternative analysis.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Handle imports gracefully for both package and standalone execution
try:
    from .importance_scoring import ImportanceScorer, ImportanceScore
    from .file_operations import read_memory_file, write_memory_file
except ImportError:
    # Fallback for standalone execution
    from importance_scoring import ImportanceScorer, ImportanceScore
    from file_operations import read_memory_file, write_memory_file

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of memory reasoning decisions."""
    STORAGE = "storage"                    # What to remember and how
    RETRIEVAL = "retrieval"               # What memories to surface
    MODIFICATION = "modification"         # When to update vs create new
    CONDENSATION = "condensation"         # How to condense memories
    PRIORITIZATION = "prioritization"    # How to prioritize memories
    CONTEXT_INTEGRATION = "context_integration"  # How to weave memories into responses
    DELETION = "deletion"                 # When to delete memories
    ORGANIZATION = "organization"         # How to organize memory structure


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning decisions."""
    VERY_HIGH = "very_high"    # 0.9-1.0
    HIGH = "high"              # 0.7-0.89
    MEDIUM = "medium"          # 0.5-0.69
    LOW = "low"                # 0.3-0.49
    VERY_LOW = "very_low"      # 0.0-0.29


@dataclass
class ReasoningStep:
    """Individual step in chain-of-thought reasoning."""
    step_number: int
    description: str
    evidence: List[str]
    confidence: float
    alternatives_considered: List[str]
    timestamp: str


@dataclass
class DecisionReasoning:
    """Complete reasoning chain for a memory decision."""
    reasoning_id: str
    reasoning_type: ReasoningType
    decision_summary: str
    reasoning_steps: List[ReasoningStep]
    overall_confidence: float
    confidence_level: ConfidenceLevel
    factors_considered: Dict[str, Any]
    alternatives_analyzed: List[Dict[str, Any]]
    expected_outcomes: List[str]
    potential_risks: List[str]
    decision_rationale: str
    context: Dict[str, Any]
    timestamp: str
    execution_time_ms: float


@dataclass
class ReasoningConfig:
    """Configuration for memory reasoning engine."""
    enable_detailed_reasoning: bool = True
    enable_alternative_analysis: bool = True
    enable_risk_assessment: bool = True
    min_confidence_threshold: float = 0.3
    max_reasoning_steps: int = 10
    reasoning_timeout_seconds: int = 5
    log_all_decisions: bool = True
    save_reasoning_history: bool = True
    reasoning_history_days: int = 30


class MemoryReasoningEngine:
    """
    Comprehensive reasoning engine for memory decisions.
    
    This engine provides detailed chain-of-thought reasoning for all memory
    operations, creating audit trails, confidence assessments, and alternative
    analysis for transparency and debugging.
    """
    
    def __init__(
        self,
        base_path: str,
        config: Optional[ReasoningConfig] = None
    ):
        """
        Initialize the memory reasoning engine.
        
        Args:
            base_path: Base path for memory storage
            config: Optional reasoning configuration
        """
        self.base_path = Path(base_path)
        self.config = config or ReasoningConfig()
        
        # Initialize importance scorer for decision support
        self.importance_scorer = ImportanceScorer()
        
        # Reasoning history storage
        self.reasoning_dir = self.base_path / 'system' / 'reasoning'
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)
        
        # Decision templates and prompts
        self.reasoning_templates = self._load_reasoning_templates()
        
        # Reasoning cache for session
        self.reasoning_cache: Dict[str, DecisionReasoning] = {}
    
    def reason_storage_decision(
        self,
        content: str,
        proposed_metadata: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> DecisionReasoning:
        """
        Generate reasoning for memory storage decisions.
        
        Args:
            content: Content to potentially store
            proposed_metadata: Proposed metadata for storage
            context: Current context for decision
            
        Returns:
            DecisionReasoning with detailed analysis
        """
        start_time = datetime.now()
        reasoning_id = f"storage_{uuid.uuid4().hex[:8]}"
        
        reasoning_steps = []
        factors_considered = {}
        alternatives = []
        
        # Step 1: Analyze content importance
        importance_score = self.importance_scorer.calculate_importance(
            content, 
            proposed_metadata.get('memory_type', 'interaction'),
            proposed_metadata,
            context
        )
        
        step1 = ReasoningStep(
            step_number=1,
            description="Analyze content importance and relevance",
            evidence=[
                f"Importance score: {importance_score.normalized_score}/10",
                f"Content length: {len(content)} characters",
                f"Memory type: {proposed_metadata.get('memory_type', 'interaction')}",
                f"Category: {proposed_metadata.get('category', 'unknown')}"
            ],
            confidence=importance_score.confidence,
            alternatives_considered=[
                "Skip storage if importance too low",
                "Store with different categorization",
                "Merge with existing similar memory"
            ],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step1)
        factors_considered['importance_analysis'] = importance_score.factor_scores
        
        # Step 2: Check for redundancy
        redundancy_analysis = self._analyze_content_redundancy(content, proposed_metadata)
        
        step2 = ReasoningStep(
            step_number=2,
            description="Check for redundant or duplicate information",
            evidence=[
                f"Similar content found: {redundancy_analysis['similar_count']} files",
                f"Redundancy score: {redundancy_analysis['redundancy_score']:.2f}",
                f"Unique information ratio: {redundancy_analysis['uniqueness_ratio']:.2f}"
            ],
            confidence=0.8,
            alternatives_considered=[
                "Merge with existing similar memory",
                "Update existing memory instead of creating new",
                "Store as separate but linked memory"
            ],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step2)
        factors_considered['redundancy_analysis'] = redundancy_analysis
        
        # Step 3: Determine storage strategy
        storage_strategy = self._determine_storage_strategy(
            importance_score, redundancy_analysis, proposed_metadata
        )
        
        step3 = ReasoningStep(
            step_number=3,
            description="Determine optimal storage strategy",
            evidence=[
                f"Recommended strategy: {storage_strategy['strategy']}",
                f"Target location: {storage_strategy['location']}",
                f"Metadata adjustments: {len(storage_strategy['metadata_changes'])} changes"
            ],
            confidence=storage_strategy['confidence'],
            alternatives_considered=storage_strategy['alternatives'],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step3)
        factors_considered['storage_strategy'] = storage_strategy
        
        # Step 4: Assess long-term value
        long_term_value = self._assess_long_term_value(content, proposed_metadata, context)
        
        step4 = ReasoningStep(
            step_number=4,
            description="Assess long-term memory value and retention needs",
            evidence=[
                f"Long-term value score: {long_term_value['value_score']:.2f}",
                f"Predicted relevance duration: {long_term_value['relevance_days']} days",
                f"Condensation eligibility: {long_term_value['condensation_eligible']}"
            ],
            confidence=long_term_value['confidence'],
            alternatives_considered=[
                "Store with expiration date",
                "Mark for early condensation",
                "Store in temporary category"
            ],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step4)
        factors_considered['long_term_value'] = long_term_value
        
        # Calculate overall confidence
        step_confidences = [step.confidence for step in reasoning_steps]
        overall_confidence = sum(step_confidences) / len(step_confidences)
        confidence_level = self._determine_confidence_level(overall_confidence)
        
        # Generate decision summary
        if importance_score.normalized_score >= 7:
            decision = "STORE - High importance content"
        elif redundancy_analysis['redundancy_score'] > 0.8:
            decision = "MERGE - Highly redundant with existing content"
        elif importance_score.normalized_score >= 4:
            decision = "STORE - Moderate importance, worth preserving"
        else:
            decision = "SKIP - Low importance, not worth storing"
        
        # Analyze alternatives
        alternatives_analyzed = [
            {
                "alternative": "Store in different category",
                "pros": ["Better organization", "Easier retrieval"],
                "cons": ["May not match user expectations"],
                "confidence": 0.6
            },
            {
                "alternative": "Merge with existing memory",
                "pros": ["Reduces redundancy", "Consolidates information"],
                "cons": ["May lose specific context", "Harder to track changes"],
                "confidence": 0.7
            },
            {
                "alternative": "Skip storage entirely",
                "pros": ["Saves storage space", "Reduces noise"],
                "cons": ["May lose valuable information", "User might expect it to be remembered"],
                "confidence": 0.4
            }
        ]
        
        # Expected outcomes and risks
        expected_outcomes = [
            f"Memory will be stored with importance score {importance_score.normalized_score}",
            f"Content will be accessible for future reference",
            f"Storage strategy: {storage_strategy['strategy']}"
        ]
        
        potential_risks = [
            "Information may become outdated over time",
            "Storage location may not be optimal for retrieval",
            "Content may be redundant with future memories"
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        reasoning = DecisionReasoning(
            reasoning_id=reasoning_id,
            reasoning_type=ReasoningType.STORAGE,
            decision_summary=decision,
            reasoning_steps=reasoning_steps,
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            factors_considered=factors_considered,
            alternatives_analyzed=alternatives_analyzed,
            expected_outcomes=expected_outcomes,
            potential_risks=potential_risks,
            decision_rationale=f"Based on importance score {importance_score.normalized_score}/10 and redundancy analysis",
            context=context or {},
            timestamp=datetime.now().isoformat(),
            execution_time_ms=execution_time
        )
        
        # Cache and optionally save reasoning
        self.reasoning_cache[reasoning_id] = reasoning
        if self.config.save_reasoning_history:
            self._save_reasoning(reasoning)
        
        return reasoning
    
    def reason_retrieval_decision(
        self,
        query_context: Dict[str, Any],
        available_memories: List[str],
        max_memories: int = 10
    ) -> DecisionReasoning:
        """
        Generate reasoning for memory retrieval decisions.
        
        Args:
            query_context: Context for the retrieval query
            available_memories: List of available memory file paths
            max_memories: Maximum number of memories to retrieve
            
        Returns:
            DecisionReasoning with retrieval analysis
        """
        start_time = datetime.now()
        reasoning_id = f"retrieval_{uuid.uuid4().hex[:8]}"
        
        reasoning_steps = []
        factors_considered = {}
        
        # Step 1: Analyze query context
        context_analysis = self._analyze_query_context(query_context)
        
        step1 = ReasoningStep(
            step_number=1,
            description="Analyze query context and information needs",
            evidence=[
                f"Query type: {context_analysis['query_type']}",
                f"Topic keywords: {', '.join(context_analysis['keywords'][:5])}",
                f"Temporal context: {context_analysis['temporal_context']}",
                f"Urgency level: {context_analysis['urgency']}"
            ],
            confidence=context_analysis['confidence'],
            alternatives_considered=[
                "Broad retrieval for comprehensive context",
                "Narrow retrieval for specific information",
                "Recent-only retrieval for current relevance"
            ],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step1)
        factors_considered['context_analysis'] = context_analysis
        
        # Step 2: Score memory relevance
        memory_scores = self._score_memory_relevance(available_memories, query_context)
        
        step2 = ReasoningStep(
            step_number=2,
            description="Score relevance of available memories",
            evidence=[
                f"Memories analyzed: {len(available_memories)}",
                f"High relevance (>0.7): {len([s for s in memory_scores if s['score'] > 0.7])}",
                f"Medium relevance (0.4-0.7): {len([s for s in memory_scores if 0.4 <= s['score'] <= 0.7])}",
                f"Top score: {max([s['score'] for s in memory_scores], default=0):.2f}"
            ],
            confidence=0.8,
            alternatives_considered=[
                "Include more lower-relevance memories for context",
                "Focus only on highest-relevance memories",
                "Include recent memories regardless of relevance"
            ],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step2)
        factors_considered['memory_scores'] = memory_scores
        
        # Step 3: Apply retrieval strategy
        retrieval_strategy = self._determine_retrieval_strategy(
            context_analysis, memory_scores, max_memories
        )
        
        step3 = ReasoningStep(
            step_number=3,
            description="Apply optimal retrieval strategy",
            evidence=[
                f"Strategy: {retrieval_strategy['strategy']}",
                f"Selected memories: {len(retrieval_strategy['selected_memories'])}",
                f"Diversity score: {retrieval_strategy['diversity_score']:.2f}",
                f"Coverage score: {retrieval_strategy['coverage_score']:.2f}"
            ],
            confidence=retrieval_strategy['confidence'],
            alternatives_considered=retrieval_strategy['alternatives'],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step3)
        factors_considered['retrieval_strategy'] = retrieval_strategy
        
        # Calculate overall confidence
        overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        confidence_level = self._determine_confidence_level(overall_confidence)
        
        # Generate decision summary
        selected_count = len(retrieval_strategy['selected_memories'])
        decision = f"RETRIEVE {selected_count} memories using {retrieval_strategy['strategy']} strategy"
        
        # Expected outcomes and risks
        expected_outcomes = [
            f"Will retrieve {selected_count} most relevant memories",
            f"Coverage of query context: {retrieval_strategy['coverage_score']:.1%}",
            f"Information diversity: {retrieval_strategy['diversity_score']:.1%}"
        ]
        
        potential_risks = [
            "May miss relevant memories not in top results",
            "Retrieved memories may be outdated",
            "Information overload if too many memories retrieved"
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        reasoning = DecisionReasoning(
            reasoning_id=reasoning_id,
            reasoning_type=ReasoningType.RETRIEVAL,
            decision_summary=decision,
            reasoning_steps=reasoning_steps,
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            factors_considered=factors_considered,
            alternatives_analyzed=[],
            expected_outcomes=expected_outcomes,
            potential_risks=potential_risks,
            decision_rationale=f"Selected {selected_count} memories based on relevance and diversity",
            context=query_context,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=execution_time
        )
        
        self.reasoning_cache[reasoning_id] = reasoning
        if self.config.save_reasoning_history:
            self._save_reasoning(reasoning)
        
        return reasoning
    
    def reason_modification_decision(
        self,
        existing_memory_path: str,
        proposed_changes: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> DecisionReasoning:
        """
        Generate reasoning for memory modification decisions.
        
        Args:
            existing_memory_path: Path to existing memory file
            proposed_changes: Proposed changes to the memory
            context: Context for the modification
            
        Returns:
            DecisionReasoning with modification analysis
        """
        start_time = datetime.now()
        reasoning_id = f"modification_{uuid.uuid4().hex[:8]}"
        
        reasoning_steps = []
        factors_considered = {}
        
        # Step 1: Analyze existing memory
        try:
            existing_frontmatter, existing_content = read_memory_file(existing_memory_path)
            memory_analysis = self._analyze_existing_memory(existing_frontmatter, existing_content)
        except Exception as e:
            logger.error(f"Error reading existing memory: {e}")
            memory_analysis = {'error': str(e), 'age_days': 0, 'importance': 5}
        
        step1 = ReasoningStep(
            step_number=1,
            description="Analyze existing memory characteristics",
            evidence=[
                f"Memory age: {memory_analysis.get('age_days', 0)} days",
                f"Current importance: {memory_analysis.get('importance', 5)}/10",
                f"Memory type: {memory_analysis.get('memory_type', 'unknown')}",
                f"Last modified: {memory_analysis.get('last_modified', 'unknown')}"
            ],
            confidence=0.9 if 'error' not in memory_analysis else 0.3,
            alternatives_considered=[
                "Create new memory instead of modifying",
                "Archive existing and create new",
                "Merge changes with existing content"
            ],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step1)
        factors_considered['memory_analysis'] = memory_analysis
        
        # Step 2: Assess change impact
        change_impact = self._assess_change_impact(
            existing_content if 'error' not in memory_analysis else "",
            proposed_changes
        )
        
        step2 = ReasoningStep(
            step_number=2,
            description="Assess impact and significance of proposed changes",
            evidence=[
                f"Change magnitude: {change_impact['magnitude']}",
                f"Content similarity: {change_impact['similarity']:.2f}",
                f"Information preservation: {change_impact['preservation_score']:.2f}",
                f"New information ratio: {change_impact['new_info_ratio']:.2f}"
            ],
            confidence=change_impact['confidence'],
            alternatives_considered=[
                "Apply changes incrementally",
                "Create versioned copy before changes",
                "Reject changes if too disruptive"
            ],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step2)
        factors_considered['change_impact'] = change_impact
        
        # Step 3: Determine modification strategy
        modification_strategy = self._determine_modification_strategy(
            memory_analysis, change_impact, proposed_changes
        )
        
        step3 = ReasoningStep(
            step_number=3,
            description="Determine optimal modification approach",
            evidence=[
                f"Recommended action: {modification_strategy['action']}",
                f"Preservation method: {modification_strategy['preservation']}",
                f"Update strategy: {modification_strategy['strategy']}"
            ],
            confidence=modification_strategy['confidence'],
            alternatives_considered=modification_strategy['alternatives'],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step3)
        factors_considered['modification_strategy'] = modification_strategy
        
        # Calculate overall confidence
        overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        confidence_level = self._determine_confidence_level(overall_confidence)
        
        # Generate decision summary
        decision = f"{modification_strategy['action']} - {modification_strategy['strategy']}"
        
        # Expected outcomes and risks
        expected_outcomes = [
            f"Memory will be {modification_strategy['action'].lower()}",
            f"Information preservation: {change_impact['preservation_score']:.1%}",
            f"Change tracking: {modification_strategy['preservation']}"
        ]
        
        potential_risks = [
            "Original information may be lost",
            "Changes may introduce inconsistencies",
            "Historical context may be disrupted"
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        reasoning = DecisionReasoning(
            reasoning_id=reasoning_id,
            reasoning_type=ReasoningType.MODIFICATION,
            decision_summary=decision,
            reasoning_steps=reasoning_steps,
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            factors_considered=factors_considered,
            alternatives_analyzed=[],
            expected_outcomes=expected_outcomes,
            potential_risks=potential_risks,
            decision_rationale=f"Modification strategy based on change impact and memory characteristics",
            context=context or {},
            timestamp=datetime.now().isoformat(),
            execution_time_ms=execution_time
        )
        
        self.reasoning_cache[reasoning_id] = reasoning
        if self.config.save_reasoning_history:
            self._save_reasoning(reasoning)
        
        return reasoning
    
    def reason_context_integration_decision(
        self,
        retrieved_memories: List[Dict[str, Any]],
        current_context: Dict[str, Any],
        response_goal: str
    ) -> DecisionReasoning:
        """
        Generate reasoning for how to integrate memories into response context.
        
        Args:
            retrieved_memories: List of retrieved memory data
            current_context: Current conversation context
            response_goal: Goal for the response
            
        Returns:
            DecisionReasoning with integration strategy
        """
        start_time = datetime.now()
        reasoning_id = f"integration_{uuid.uuid4().hex[:8]}"
        
        reasoning_steps = []
        factors_considered = {}
        
        # Step 1: Analyze memory relevance to current context
        relevance_analysis = self._analyze_memory_context_relevance(
            retrieved_memories, current_context
        )
        
        step1 = ReasoningStep(
            step_number=1,
            description="Analyze memory relevance to current conversation context",
            evidence=[
                f"Memories analyzed: {len(retrieved_memories)}",
                f"High relevance: {relevance_analysis['high_relevance_count']}",
                f"Average relevance: {relevance_analysis['avg_relevance']:.2f}",
                f"Context overlap: {relevance_analysis['context_overlap']:.2f}"
            ],
            confidence=relevance_analysis['confidence'],
            alternatives_considered=[
                "Use only highest relevance memories",
                "Include broader context for completeness",
                "Focus on recent memories only"
            ],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step1)
        factors_considered['relevance_analysis'] = relevance_analysis
        
        # Step 2: Determine integration strategy
        integration_strategy = self._determine_integration_strategy(
            retrieved_memories, current_context, response_goal
        )
        
        step2 = ReasoningStep(
            step_number=2,
            description="Determine optimal memory integration strategy",
            evidence=[
                f"Integration approach: {integration_strategy['approach']}",
                f"Memory weaving style: {integration_strategy['style']}",
                f"Reference explicitness: {integration_strategy['explicitness']}",
                f"Memories to integrate: {len(integration_strategy['selected_memories'])}"
            ],
            confidence=integration_strategy['confidence'],
            alternatives_considered=integration_strategy['alternatives'],
            timestamp=datetime.now().isoformat()
        )
        reasoning_steps.append(step2)
        factors_considered['integration_strategy'] = integration_strategy
        
        # Calculate overall confidence
        overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        confidence_level = self._determine_confidence_level(overall_confidence)
        
        # Generate decision summary
        selected_count = len(integration_strategy['selected_memories'])
        decision = f"INTEGRATE {selected_count} memories using {integration_strategy['approach']} approach"
        
        # Expected outcomes and risks
        expected_outcomes = [
            f"Will integrate {selected_count} relevant memories",
            f"Integration style: {integration_strategy['style']}",
            f"Context continuity maintained"
        ]
        
        potential_risks = [
            "May overwhelm response with too much context",
            "Memories may not align perfectly with current topic",
            "Integration may feel forced or unnatural"
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        reasoning = DecisionReasoning(
            reasoning_id=reasoning_id,
            reasoning_type=ReasoningType.CONTEXT_INTEGRATION,
            decision_summary=decision,
            reasoning_steps=reasoning_steps,
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            factors_considered=factors_considered,
            alternatives_analyzed=[],
            expected_outcomes=expected_outcomes,
            potential_risks=potential_risks,
            decision_rationale=f"Integration strategy optimized for {response_goal}",
            context=current_context,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=execution_time
        )
        
        self.reasoning_cache[reasoning_id] = reasoning
        if self.config.save_reasoning_history:
            self._save_reasoning(reasoning)
        
        return reasoning
    
    def get_reasoning_summary(self, reasoning_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of reasoning for a specific decision."""
        if reasoning_id in self.reasoning_cache:
            reasoning = self.reasoning_cache[reasoning_id]
            return {
                'id': reasoning.reasoning_id,
                'type': reasoning.reasoning_type.value,
                'decision': reasoning.decision_summary,
                'confidence': reasoning.overall_confidence,
                'confidence_level': reasoning.confidence_level.value,
                'steps': len(reasoning.reasoning_steps),
                'timestamp': reasoning.timestamp,
                'execution_time_ms': reasoning.execution_time_ms
            }
        return None
    
    def get_reasoning_history(
        self,
        reasoning_type: Optional[ReasoningType] = None,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get reasoning history for analysis."""
        cutoff_date = datetime.now() - timedelta(days=days)
        history = []
        
        # Load from saved reasoning files
        for reasoning_file in self.reasoning_dir.glob("*.json"):
            try:
                with open(reasoning_file, 'r') as f:
                    reasoning_data = json.load(f)
                    
                reasoning_date = datetime.fromisoformat(reasoning_data['timestamp'])
                if reasoning_date >= cutoff_date:
                    if reasoning_type is None or reasoning_data['reasoning_type'] == reasoning_type.value:
                        history.append({
                            'id': reasoning_data['reasoning_id'],
                            'type': reasoning_data['reasoning_type'],
                            'decision': reasoning_data['decision_summary'],
                            'confidence': reasoning_data['overall_confidence'],
                            'timestamp': reasoning_data['timestamp']
                        })
            except Exception as e:
                logger.warning(f"Error loading reasoning file {reasoning_file}: {e}")
        
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)
    
    # Helper methods for analysis
    
    def _analyze_content_redundancy(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze content for redundancy with existing memories."""
        # Simplified redundancy analysis
        # In a real implementation, this would use semantic similarity
        
        similar_count = 0
        redundancy_score = 0.0
        uniqueness_ratio = 1.0
        
        # Basic keyword-based similarity check
        content_words = set(content.lower().split())
        
        # This would normally check against existing memories
        # For now, return mock analysis
        
        return {
            'similar_count': similar_count,
            'redundancy_score': redundancy_score,
            'uniqueness_ratio': uniqueness_ratio,
            'analysis_method': 'keyword_based'
        }
    
    def _determine_storage_strategy(
        self,
        importance_score: ImportanceScore,
        redundancy_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine optimal storage strategy."""
        
        if importance_score.normalized_score >= 8:
            strategy = "high_priority_storage"
            location = "core"
        elif redundancy_analysis['redundancy_score'] > 0.7:
            strategy = "merge_with_existing"
            location = "interactions"
        else:
            strategy = "standard_storage"
            location = "interactions"
        
        return {
            'strategy': strategy,
            'location': location,
            'metadata_changes': [],
            'confidence': 0.8,
            'alternatives': [
                "Store in different category",
                "Apply different importance threshold",
                "Use time-based storage location"
            ]
        }
    
    def _assess_long_term_value(
        self,
        content: str,
        metadata: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess long-term value of memory."""
        
        # Simple heuristic-based assessment
        value_score = 0.5
        relevance_days = 30
        condensation_eligible = True
        
        # Adjust based on content characteristics
        if any(keyword in content.lower() for keyword in ['important', 'remember', 'crucial']):
            value_score += 0.3
            relevance_days = 90
        
        if metadata.get('memory_type') == 'user_profile':
            value_score += 0.4
            relevance_days = 365
            condensation_eligible = False
        
        return {
            'value_score': min(value_score, 1.0),
            'relevance_days': relevance_days,
            'condensation_eligible': condensation_eligible,
            'confidence': 0.7
        }
    
    def _analyze_query_context(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query context for retrieval decisions."""
        
        # Extract key information from query context
        query_type = query_context.get('type', 'general')
        keywords = query_context.get('keywords', [])
        temporal_context = query_context.get('temporal', 'current')
        urgency = query_context.get('urgency', 'normal')
        
        return {
            'query_type': query_type,
            'keywords': keywords,
            'temporal_context': temporal_context,
            'urgency': urgency,
            'confidence': 0.8
        }
    
    def _score_memory_relevance(
        self,
        memory_paths: List[str],
        query_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score relevance of memories to query context."""
        
        scores = []
        for path in memory_paths:
            # Simple scoring based on file path and context
            # In real implementation, would analyze content
            
            base_score = 0.5
            
            # Boost score for certain patterns
            if 'user_profile' in path:
                base_score += 0.3
            if any(keyword in path.lower() for keyword in query_context.get('keywords', [])):
                base_score += 0.2
            
            scores.append({
                'path': path,
                'score': min(base_score, 1.0),
                'reasoning': 'Path-based scoring'
            })
        
        return sorted(scores, key=lambda x: x['score'], reverse=True)
    
    def _determine_retrieval_strategy(
        self,
        context_analysis: Dict[str, Any],
        memory_scores: List[Dict[str, Any]],
        max_memories: int
    ) -> Dict[str, Any]:
        """Determine optimal retrieval strategy."""
        
        # Select top-scoring memories
        selected_memories = memory_scores[:max_memories]
        
        strategy = "relevance_based"
        diversity_score = 0.7  # Mock diversity calculation
        coverage_score = 0.8   # Mock coverage calculation
        
        return {
            'strategy': strategy,
            'selected_memories': selected_memories,
            'diversity_score': diversity_score,
            'coverage_score': coverage_score,
            'confidence': 0.8,
            'alternatives': [
                "Time-based selection",
                "Category-balanced selection",
                "Importance-weighted selection"
            ]
        }
    
    def _analyze_existing_memory(
        self,
        frontmatter: Dict[str, Any],
        content: str
    ) -> Dict[str, Any]:
        """Analyze characteristics of existing memory."""
        
        created_str = frontmatter.get('created', '')
        age_days = 0
        
        if created_str:
            try:
                created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                age_days = (datetime.now() - created_date).days
            except Exception:
                pass
        
        return {
            'age_days': age_days,
            'importance': frontmatter.get('importance_score', 5),
            'memory_type': frontmatter.get('memory_type', 'interaction'),
            'last_modified': frontmatter.get('last_updated', 'unknown'),
            'content_length': len(content)
        }
    
    def _assess_change_impact(
        self,
        existing_content: str,
        proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess impact of proposed changes."""
        
        # Simple change impact analysis
        new_content = proposed_changes.get('content', existing_content)
        
        # Calculate similarity (simplified)
        similarity = 0.8 if existing_content else 0.0
        preservation_score = 0.9
        new_info_ratio = 0.2
        magnitude = "moderate"
        
        return {
            'magnitude': magnitude,
            'similarity': similarity,
            'preservation_score': preservation_score,
            'new_info_ratio': new_info_ratio,
            'confidence': 0.7
        }
    
    def _determine_modification_strategy(
        self,
        memory_analysis: Dict[str, Any],
        change_impact: Dict[str, Any],
        proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine optimal modification strategy."""
        
        if change_impact['magnitude'] == "major":
            action = "CREATE_NEW"
            strategy = "preserve_original"
        elif memory_analysis.get('importance', 5) >= 8:
            action = "UPDATE_WITH_HISTORY"
            strategy = "versioned_update"
        else:
            action = "UPDATE"
            strategy = "direct_update"
        
        return {
            'action': action,
            'strategy': strategy,
            'preservation': 'timestamp_tracking',
            'confidence': 0.8,
            'alternatives': [
                "Archive original and create new",
                "Merge changes gradually",
                "Create linked memory instead"
            ]
        }
    
    def _analyze_memory_context_relevance(
        self,
        memories: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze relevance of memories to current context."""
        
        high_relevance_count = len([m for m in memories if m.get('relevance_score', 0.5) > 0.7])
        avg_relevance = sum(m.get('relevance_score', 0.5) for m in memories) / len(memories) if memories else 0
        context_overlap = 0.6  # Mock calculation
        
        return {
            'high_relevance_count': high_relevance_count,
            'avg_relevance': avg_relevance,
            'context_overlap': context_overlap,
            'confidence': 0.8
        }
    
    def _determine_integration_strategy(
        self,
        memories: List[Dict[str, Any]],
        context: Dict[str, Any],
        response_goal: str
    ) -> Dict[str, Any]:
        """Determine optimal memory integration strategy."""
        
        # Select most relevant memories
        selected_memories = [m for m in memories if m.get('relevance_score', 0.5) > 0.6]
        
        approach = "contextual_weaving"
        style = "natural_references"
        explicitness = "subtle"
        
        return {
            'approach': approach,
            'style': style,
            'explicitness': explicitness,
            'selected_memories': selected_memories,
            'confidence': 0.8,
            'alternatives': [
                "Explicit memory citations",
                "Background context only",
                "Direct memory quotes"
            ]
        }
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level."""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _load_reasoning_templates(self) -> Dict[str, Any]:
        """Load reasoning templates for different decision types."""
        # This would load from configuration files
        return {
            'storage': {},
            'retrieval': {},
            'modification': {},
            'integration': {}
        }
    
    def _save_reasoning(self, reasoning: DecisionReasoning) -> None:
        """Save reasoning to persistent storage."""
        try:
            reasoning_file = self.reasoning_dir / f"{reasoning.reasoning_id}.json"
            with open(reasoning_file, 'w') as f:
                json.dump(asdict(reasoning), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving reasoning {reasoning.reasoning_id}: {e}")
    
    def cleanup_old_reasoning(self, days: int = 30) -> int:
        """Clean up old reasoning files."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for reasoning_file in self.reasoning_dir.glob("*.json"):
            try:
                if reasoning_file.stat().st_mtime < cutoff_date.timestamp():
                    reasoning_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Error cleaning up reasoning file {reasoning_file}: {e}")
        
        return cleaned_count 