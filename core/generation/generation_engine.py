"""
Complete Generation Engine - Orchestrates all generation components
Integrates LLM, prompt building, citation formatting, and validation

File: core/generation/generation_engine.py
"""

from typing import Dict, Any, List, Optional, Generator
import time
from logger_utils import get_logger
from .llm_engine import LLMEngine
from .prompt_builder import PromptBuilder
from .citation_formatter import CitationFormatter
from .response_validator import ResponseValidator


class GenerationEngine:
    """
    Complete generation pipeline orchestrator
    Coordinates all generation components for end-to-end response generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("GenerationEngine")
        self.config = config
        
        # Initialize components
        self.llm_engine = LLMEngine(config)
        self.prompt_builder = PromptBuilder(config)
        self.citation_formatter = CitationFormatter(config)
        self.response_validator = ResponseValidator(config)
        
        # Generation settings
        self.enable_validation = config.get('enable_validation', True)
        self.enable_enhancement = config.get('enable_enhancement', True)
        self.strict_validation = config.get('strict_validation', False)
        
        self.logger.info("GenerationEngine initialized", {
            "validation_enabled": self.enable_validation,
            "enhancement_enabled": self.enable_enhancement
        })
    
    def initialize(self) -> bool:
        """
        Initialize generation engine (load models)
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Initializing generation engine...")
        
        if not self.llm_engine.load_model():
            self.logger.error("Failed to load LLM model")
            return False
        
        self.logger.success("Generation engine initialized successfully")
        return True
    
    def generate_answer(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        thinking_mode: str = 'low'
    ) -> Dict[str, Any]:
        """
        Generate complete answer with all enhancements

        Args:
            query: User query
            retrieved_results: Retrieved and ranked documents
            query_analysis: Optional query analysis
            conversation_history: Optional conversation context
            stream: Whether to stream response
            thinking_mode: Thinking mode ('low', 'medium', 'high')

        Returns:
            Complete generation result dictionary
        """
        self.logger.info("Starting answer generation", {
            "query": query[:50] + "..." if len(query) > 50 else query,
            "num_results": len(retrieved_results),
            "stream": stream,
            "thinking_mode": thinking_mode
        })

        start_time = time.time()

        try:
            # Step 1: Determine template type
            template_type = self._determine_template_type(query, query_analysis)

            self.logger.debug("Template type determined", {
                "template": template_type
            })

            # Step 2: Build prompt with thinking mode
            prompt = self.prompt_builder.build_prompt(
                query=query,
                retrieved_results=retrieved_results,
                query_analysis=query_analysis,
                conversation_history=conversation_history,
                template_type=template_type,
                thinking_mode=thinking_mode
            )
            
            self.logger.debug("Prompt built", {
                "prompt_length": len(prompt)
            })

            # Print complete prompt for test transparency
            print("\n" + "=" * 100)
            print("COMPLETE LLM INPUT PROMPT (FULL TRANSPARENCY)")
            print("=" * 100)
            print(f"Character Count: {len(prompt):,}")
            print("-" * 100)
            print(prompt)
            print("-" * 100)
            print()

            # Step 2.5: Determine max_new_tokens based on thinking mode
            max_new_tokens = self._get_max_tokens_for_thinking_mode(thinking_mode)

            self.logger.info("Max tokens determined", {
                "thinking_mode": thinking_mode,
                "max_new_tokens": max_new_tokens
            })

            # Step 3: Generate response
            if stream:
                return self._generate_streaming_answer(
                    query=query,
                    prompt=prompt,
                    retrieved_results=retrieved_results,
                    max_new_tokens=max_new_tokens
                )
            else:
                generation_result = self.llm_engine.generate(prompt, max_new_tokens=max_new_tokens)
                
                if not generation_result['success']:
                    self.logger.error("Generation failed", {
                        "error": generation_result.get('error')
                    })
                    return {
                        'success': False,
                        'error': generation_result.get('error'),
                        'answer': '',
                        'metadata': {}
                    }
                
                raw_answer = generation_result['generated_text']

                # Step 4: Extract thinking and answer separately
                thinking, answer_only = self._extract_thinking(raw_answer)

                # Step 4.5: Handle incomplete generation (only thinking, no answer)
                if not answer_only or len(answer_only.strip()) < 20:
                    self.logger.warning("Incomplete generation detected, providing fallback response", {
                        "answer_length": len(answer_only) if answer_only else 0
                    })
                    # Provide a helpful fallback message
                    answer_only = (
                        "Mohon maaf, saya tidak dapat memberikan jawaban yang lengkap berdasarkan dokumen yang tersedia. "
                        "Silakan lihat sumber hukum di bawah untuk informasi lebih lanjut, atau coba ajukan pertanyaan dengan cara yang berbeda."
                    )

                # Step 5: Post-process response
                processed_answer = self._post_process_response(answer_only)

                # Step 6: Add citations
                cited_answer = self.citation_formatter.format_inline_references(
                    processed_answer,
                    retrieved_results
                )
                
                # Step 6: Validate response
                validation_result = None
                if self.enable_validation:
                    validation_result = self.response_validator.validate_response(
                        response=cited_answer,
                        query=query,
                        retrieved_results=retrieved_results,
                        strict=self.strict_validation
                    )
                    
                    if self.enable_enhancement:
                        cited_answer = validation_result['enhanced_response']
                
                # Step 8: Build complete result
                total_time = time.time() - start_time

                result = {
                    'success': True,
                    'answer': cited_answer,
                    'thinking': thinking,  # Preserved thinking process
                    'raw_answer': raw_answer,
                    'metadata': {
                        'query': query,
                        'template_type': template_type,
                        'num_source_docs': len(retrieved_results),
                        'generation_time': generation_result['generation_time'],
                        'total_time': total_time,
                        'tokens_generated': generation_result['tokens_generated'],
                        'tokens_per_second': generation_result['tokens_per_second'],
                        'validation': validation_result if validation_result else {},
                        'prompt_length': len(prompt),
                        'complete_prompt': prompt  # Store complete prompt for transparency
                    },
                    'citations': self._extract_citations(retrieved_results),
                    'sources': self._format_sources(retrieved_results)
                }
                
                self.logger.success("Answer generation completed", {
                    "total_time": f"{total_time:.2f}s",
                    "answer_length": len(cited_answer),
                    "tokens_generated": generation_result['tokens_generated']
                })
                
                return result
        
        except Exception as e:
            self.logger.error("Answer generation failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()[:500]
            })
            
            return {
                'success': False,
                'error': str(e),
                'answer': '',
                'metadata': {}
            }
    
    def _generate_streaming_answer(
        self,
        query: str,
        prompt: str,
        retrieved_results: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate answer with streaming"""

        self.logger.info("Starting streaming generation", {
            "max_new_tokens": max_new_tokens
        })

        full_response = ""
        tokens_generated = 0

        try:
            for chunk in self.llm_engine.generate_stream(prompt, max_new_tokens=max_new_tokens):
                if chunk['success']:
                    if not chunk['done']:
                        token = chunk['token']
                        full_response += token
                        tokens_generated = chunk['tokens_generated']
                        
                        yield {
                            'type': 'token',
                            'token': token,
                            'tokens_generated': tokens_generated,
                            'done': False
                        }
                    else:
                        # Final chunk
                        # Extract thinking BEFORE post-processing
                        thinking, answer_only = self._extract_thinking(full_response)

                        # Post-process and validate
                        processed = self._post_process_response(answer_only if answer_only else full_response)

                        cited = self.citation_formatter.format_inline_references(
                            processed,
                            retrieved_results
                        )

                        validation_result = None
                        if self.enable_validation:
                            validation_result = self.response_validator.validate_response(
                                response=cited,
                                query=query,
                                retrieved_results=retrieved_results,
                                strict=False  # Don't be strict in streaming
                            )

                            if self.enable_enhancement:
                                cited = validation_result['enhanced_response']

                        yield {
                            'type': 'complete',
                            'answer': cited,
                            'thinking': thinking,  # Include extracted thinking process
                            'raw_answer': full_response,
                            'tokens_generated': tokens_generated,
                            'generation_time': chunk.get('generation_time', 0),
                            'tokens_per_second': chunk.get('tokens_per_second', 0),
                            'validation': validation_result,
                            'citations': self._extract_citations(retrieved_results),
                            'complete_prompt': prompt,  # Store complete prompt for transparency
                            'done': True
                        }
                else:
                    yield {
                        'type': 'error',
                        'error': chunk.get('error', 'Unknown error'),
                        'done': True
                    }
                    
        except Exception as e:
            self.logger.error("Streaming generation failed", {
                "error": str(e)
            })
            yield {
                'type': 'error',
                'error': str(e),
                'done': True
            }

    def _get_max_tokens_for_thinking_mode(self, thinking_mode: str) -> int:
        """
        Menentukan max_new_tokens berdasarkan thinking mode.

        Args:
            thinking_mode: Mode berpikir ('low', 'medium', 'high')

        Returns:
            Max tokens untuk generation
        """
        thinking_mode_lower = thinking_mode.lower()

        # Mapping thinking mode ke max_new_tokens
        mode_to_tokens = {
            'low': self.config.get('max_new_tokens', 2048),  # Gunakan default dari config
            'medium': 8192,   # Medium mode: 8K tokens
            'high': 16384     # High mode: 16K tokens
        }

        max_tokens = mode_to_tokens.get(thinking_mode_lower, 2048)

        self.logger.debug("Thinking mode max tokens mapping", {
            "thinking_mode": thinking_mode,
            "max_new_tokens": max_tokens
        })

        return max_tokens

    def _determine_template_type(
        self,
        query: str,
        query_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """Determine appropriate prompt template"""
        
        if not query_analysis:
            return 'rag_qa'
        
        query_type = query_analysis.get('query_type', 'general')
        
        # Map query types to templates
        template_mapping = {
            'procedural': 'procedural',
            'specific_article': 'rag_qa',
            'definitional': 'rag_qa',
            'sanctions': 'rag_qa',
            'general': 'rag_qa'
        }
        
        # Check for follow-up or clarification
        if query_analysis.get('is_followup'):
            return 'followup'
        
        if query_analysis.get('is_clarification'):
            return 'clarification'
        
        return template_mapping.get(query_type, 'rag_qa')
    
    def _post_process_response(self, response: str) -> str:
        """Post-process generated response - preserve thinking for separate display"""

        # Sanitize
        processed = self.response_validator.sanitize_response(response)

        # Clean up formatting
        processed = processed.strip()

        return processed

    def _extract_thinking(self, response: str) -> tuple:
        """Extract thinking process from response - FIXED: More robust parsing

        Returns:
            Tuple of (thinking_content, answer_content)
        """
        import re

        # FIXED: More robust XML-like tag extraction with proper handling
        # Match opening tag, capture content, match closing tag (non-greedy, case-insensitive)
        think_pattern = r'<think\s*>(.*?)</think\s*>'

        thinking = ''
        answer = response

        try:
            # Find all thinking blocks (in case there are multiple)
            thinking_matches = re.findall(think_pattern, response, flags=re.DOTALL | re.IGNORECASE)
            if thinking_matches:
                # Combine all thinking blocks
                thinking = '\n\n'.join(match.strip() for match in thinking_matches)

            # Remove all thinking tags from answer (more conservative removal)
            answer = re.sub(think_pattern, '', response, flags=re.DOTALL | re.IGNORECASE).strip()

        except Exception as e:
            # FIXED: Graceful fallback if regex fails
            self.logger.warning("Failed to extract thinking tags with regex", {
                "error": str(e)
            })
            # Try simple string-based extraction as fallback
            if '<think>' in response.lower() and '</think>' in response.lower():
                try:
                    start_idx = response.lower().index('<think>')
                    end_idx = response.lower().index('</think>') + len('</think>')
                    thinking = response[start_idx+7:end_idx-8].strip()  # Extract between tags
                    answer = response[:start_idx] + response[end_idx:]
                    answer = answer.strip()
                except Exception:
                    # Complete fallback: treat entire response as answer
                    thinking = ''
                    answer = response

        # Also detect untagged thinking patterns (numbered steps at the start)
        # Pattern: Langkah/Step followed by number and colon at the beginning of lines
        untagged_thinking_pattern = r'^(?:Langkah|Langka|Step)\s*\d+\s*:\s*.+?(?=\n|$)'

        # Find all untagged thinking lines
        untagged_matches = re.findall(untagged_thinking_pattern, answer, flags=re.MULTILINE | re.IGNORECASE)

        if untagged_matches:
            # Check if the entire response is just thinking steps (no actual answer)
            # This happens when model outputs planning without actual content
            remaining_after_removal = re.sub(untagged_thinking_pattern, '', answer, flags=re.MULTILINE | re.IGNORECASE).strip()

            # If what remains is very short or empty, the model failed to generate actual answer
            if len(remaining_after_removal) < 50:
                # Add untagged thinking to thinking section
                if thinking:
                    thinking = thinking + '\n\n' + '\n'.join(untagged_matches)
                else:
                    thinking = '\n'.join(untagged_matches)

                # Log warning about incomplete generation
                self.logger.warning("LLM generated only thinking steps without actual answer", {
                    "thinking_steps": len(untagged_matches),
                    "remaining_length": len(remaining_after_removal)
                })

                # Return empty answer to trigger fallback
                answer = remaining_after_removal
            else:
                # There's actual content after the thinking steps, just remove the steps
                answer = remaining_after_removal

        return thinking, answer
    
    def _extract_citations(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract citation information from results with complete metadata"""

        citations = []

        for idx, result in enumerate(results, 1):
            record = result.get('record', {})

            citation = {
                'id': idx,
                'regulation_type': record.get('regulation_type', ''),
                'regulation_number': record.get('regulation_number', ''),
                'year': record.get('year', ''),
                'about': record.get('about', ''),
                'article': record.get('article', ''),
                'chapter': record.get('chapter', ''),
                'enacting_body': record.get('enacting_body', ''),
                'citation_text': self.citation_formatter.format_citation(
                    record,
                    style='standard'
                ),
                # Include all scores for detailed display
                'final_score': result.get('final_score', result.get('rerank_score', 0)),
                'rerank_score': result.get('rerank_score', 0),
                'composite_score': result.get('composite_score', 0),
                'kg_score': result.get('kg_score', 0),
                'semantic_score': result.get('semantic_score', 0),
                'keyword_score': result.get('keyword_score', 0),
                # KG metadata
                'kg_primary_domain': record.get('kg_primary_domain', ''),
                'kg_hierarchy_level': record.get('kg_hierarchy_level', 0),
                'kg_authority_score': record.get('kg_authority_score', 0),
                'kg_cross_ref_count': record.get('kg_cross_ref_count', 0),
                'kg_pagerank': record.get('kg_pagerank', 0),
                'kg_connectivity_score': record.get('kg_connectivity_score', 0),
                # Team consensus
                'team_consensus': result.get('team_consensus', False),
                'researcher_agreement': result.get('researcher_agreement', 0),
                'supporting_researchers': result.get('supporting_researchers', []),
                # Devil's advocate
                'devils_advocate_challenged': result.get('devils_advocate_challenged', False),
                'challenge_points': result.get('challenge_points', []),
                # Content
                'content': record.get('content', ''),
                # Pass through the full result for additional metadata
                'record': record
            }

            citations.append(citation)

        return citations
    
    def _format_sources(
        self,
        results: List[Dict[str, Any]],
        max_sources: int = 5
    ) -> List[Dict[str, Any]]:
        """Format source information"""
        
        sources = []
        
        for idx, result in enumerate(results[:max_sources], 1):
            record = result.get('record', {})
            
            source = {
                'id': idx,
                'title': self.citation_formatter.format_citation(
                    record,
                    style='short'
                ),
                'type': record.get('regulation_type', ''),
                'year': record.get('year', ''),
                'relevance_score': result.get(
                    'final_score',
                    result.get('rerank_score', 0)
                ),
                'url': record.get('url', ''),  # If available
                'excerpt': record.get('content', '')[:200] + "..."
            }
            
            sources.append(source)
        
        return sources
    
    def generate_follow_up_suggestions(
        self,
        query: str,
        answer: str,
        retrieved_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate follow-up question suggestions
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_results: Retrieved documents
            
        Returns:
            List of suggested follow-up questions
        """
        self.logger.info("Generating follow-up suggestions")
        
        suggestions = []
        
        # Extract key topics from results
        topics = set()
        for result in retrieved_results[:3]:
            record = result.get('record', {})
            
            # Add regulation type
            reg_type = record.get('regulation_type', '')
            if reg_type:
                topics.add(reg_type)
            
            # Extract keywords from about
            about = record.get('about', '')
            if about:
                # Simple keyword extraction
                words = about.lower().split()
                topics.update(w for w in words if len(w) > 5)
        
        # Generate template suggestions
        if 'sanksi' in query.lower() or 'pidana' in query.lower():
            suggestions.append("Apa saja prosedur penegakan hukum untuk pelanggaran ini?")
            suggestions.append("Bagaimana cara melaporkan pelanggaran?")
        
        if 'prosedur' in query.lower() or 'tata cara' in query.lower():
            suggestions.append("Apa persyaratan yang harus dipenuhi?")
            suggestions.append("Berapa lama proses ini biasanya memakan waktu?")
        
        # Add regulation-specific suggestions
        if len(retrieved_results) > 0:
            first_result = retrieved_results[0].get('record', {})
            reg_type = first_result.get('regulation_type', '')
            reg_num = first_result.get('regulation_number', '')
            
            if reg_type and reg_num:
                suggestions.append(
                    f"Apa peraturan lain yang berkaitan dengan {reg_type} No. {reg_num}?"
                )
        
        return suggestions[:3]  # Return top 3
    
    def shutdown(self):
        """Shutdown generation engine and cleanup"""
        self.logger.info("Shutting down generation engine")
        self.llm_engine.unload_model()
        self.logger.success("Generation engine shut down")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        return {
            'llm_info': self.llm_engine.get_model_info(),
            'prompt_info': self.prompt_builder.get_template_info(),
            'config': {
                'validation_enabled': self.enable_validation,
                'enhancement_enabled': self.enable_enhancement,
                'strict_validation': self.strict_validation
            }
        }