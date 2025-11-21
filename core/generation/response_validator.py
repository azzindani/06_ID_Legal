"""
Response Validator for Indonesian Legal RAG System
Validates and enhances generated responses for quality and safety

File: core/generation/response_validator.py
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from logger_utils import get_logger


class ResponseValidator:
    """
    Validates generated responses for quality, safety, and legal accuracy
    Performs post-generation checks and enhancements
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("ResponseValidator")
        self.config = config
        
        # Validation thresholds
        self.min_length = config.get('min_response_length', 50)
        self.max_length = config.get('max_response_length', 4000)
        self.require_citations = config.get('require_citations', True)
        
        # Quality checks
        self.quality_checks = [
            self._check_length,
            self._check_citations,
            self._check_completeness,
            self._check_disclaimers,
            self._check_coherence
        ]
        
        # Safety patterns
        self.unsafe_patterns = [
            r'saya tidak tahu',
            r'tidak ada informasi',
            r'maaf.*tidak bisa',
            r'data tidak tersedia'
        ]
        
        self.logger.info("ResponseValidator initialized", {
            "min_length": self.min_length,
            "require_citations": self.require_citations
        })
    
    def validate_response(
        self,
        response: str,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate generated response
        
        Args:
            response: Generated response text
            query: Original query
            retrieved_results: Retrieved documents
            strict: Whether to use strict validation
            
        Returns:
            Validation result dictionary
        """
        self.logger.info("Validating response", {
            "response_length": len(response),
            "strict": strict
        })
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 1.0,
            'checks_passed': [],
            'checks_failed': [],
            'enhanced_response': response
        }
        
        # Run quality checks
        for check in self.quality_checks:
            check_name = check.__name__
            
            try:
                passed, message, score_impact = check(
                    response,
                    query,
                    retrieved_results
                )
                
                if passed:
                    validation_result['checks_passed'].append(check_name)
                else:
                    validation_result['checks_failed'].append(check_name)
                    
                    if strict:
                        validation_result['errors'].append(message)
                        validation_result['is_valid'] = False
                    else:
                        validation_result['warnings'].append(message)
                    
                    validation_result['quality_score'] *= (1.0 - score_impact)
                
            except Exception as e:
                self.logger.error(f"Check {check_name} failed", {
                    "error": str(e)
                })
                validation_result['warnings'].append(
                    f"Check {check_name} encountered error: {str(e)}"
                )
        
        # Apply enhancements if valid
        if validation_result['is_valid'] or not strict:
            validation_result['enhanced_response'] = self._enhance_response(
                response,
                retrieved_results
            )
        
        self.logger.info("Validation completed", {
            "is_valid": validation_result['is_valid'],
            "quality_score": f"{validation_result['quality_score']:.2f}",
            "checks_passed": len(validation_result['checks_passed']),
            "checks_failed": len(validation_result['checks_failed'])
        })
        
        return validation_result
    
    def _check_length(
        self,
        response: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float]:
        """Check if response length is appropriate"""
        
        length = len(response.strip())
        
        if length < self.min_length:
            return False, f"Response too short ({length} chars, min: {self.min_length})", 0.3
        
        if length > self.max_length:
            return False, f"Response too long ({length} chars, max: {self.max_length})", 0.1
        
        return True, "Length check passed", 0.0
    
    def _check_citations(
        self,
        response: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float]:
        """Check if response includes proper citations"""
        
        if not self.require_citations:
            return True, "Citations not required", 0.0
        
        # Look for citation patterns
        citation_patterns = [
            r'\[Dokumen \d+\]',
            r'\[Dok\. \d+\]',
            r'(?:UU|PP|Perpres|Permen).*?(?:No\.|Nomor)\s*\d+',
            r'Pasal \d+',
            r'berdasarkan.*?(?:peraturan|undang-undang)'
        ]
        
        has_citations = False
        for pattern in citation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                has_citations = True
                break
        
        if not has_citations and len(results) > 0:
            return False, "Response lacks proper citations", 0.2
        
        return True, "Citations check passed", 0.0
    
    def _check_completeness(
        self,
        response: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float]:
        """Check if response addresses the query completely"""
        
        # Extract key terms from query
        query_lower = query.lower()
        
        # Check for question words and their answers
        question_indicators = {
            'apa': ['adalah', 'merupakan', 'yaitu'],
            'siapa': ['adalah', 'merupakan'],
            'kapan': ['tanggal', 'waktu', 'pada'],
            'dimana': ['di', 'pada', 'lokasi'],
            'bagaimana': ['cara', 'prosedur', 'dengan'],
            'berapa': ['jumlah', 'sebanyak', 'sejumlah'],
            'mengapa': ['karena', 'sebab', 'alasan']
        }
        
        response_lower = response.lower()
        
        for question_word, answer_words in question_indicators.items():
            if question_word in query_lower:
                # Check if response has appropriate answer indicators
                has_answer = any(word in response_lower for word in answer_words)
                if not has_answer:
                    return False, f"Response may not fully address '{question_word}' question", 0.15
        
        # Check for unsafe patterns indicating incomplete answers
        for pattern in self.unsafe_patterns:
            if re.search(pattern, response_lower):
                return False, "Response contains uncertainty indicators", 0.2
        
        return True, "Completeness check passed", 0.0
    
    def _check_disclaimers(
        self,
        response: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float]:
        """Check if response includes appropriate legal disclaimers"""
        
        # Legal advice indicators
        advice_indicators = [
            'sanksi',
            'pidana',
            'denda',
            'hukuman',
            'gugatan',
            'tuntutan',
            'perdata'
        ]
        
        query_lower = query.lower()
        needs_disclaimer = any(indicator in query_lower for indicator in advice_indicators)
        
        if needs_disclaimer:
            disclaimer_patterns = [
                r'konsultasi.*?(?:ahli hukum|pengacara|advokat)',
                r'rekomendasi.*?(?:ahli hukum|pengacara)',
                r'sebaiknya.*?konsultasi',
                r'disarankan.*?(?:ahli hukum|pengacara)'
            ]
            
            has_disclaimer = any(
                re.search(pattern, response, re.IGNORECASE)
                for pattern in disclaimer_patterns
            )
            
            if not has_disclaimer:
                return False, "Response lacks legal consultation disclaimer", 0.1
        
        return True, "Disclaimer check passed", 0.0
    
    def _check_coherence(
        self,
        response: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float]:
        """Check response coherence and structure"""
        
        # Check for repeated phrases
        sentences = response.split('.')
        if len(sentences) > 3:
            # Count repeated sentences
            unique_sentences = set(s.strip().lower() for s in sentences if len(s.strip()) > 20)
            if len(unique_sentences) < len(sentences) * 0.7:
                return False, "Response contains excessive repetition", 0.15
        
        # Check for abrupt endings
        if not response.strip().endswith(('.', '!', '?', '"')):
            return False, "Response has incomplete ending", 0.1
        
        # Check for proper paragraph structure
        paragraphs = response.split('\n\n')
        if len(paragraphs) == 1 and len(response) > 500:
            # Long response without paragraphs
            return False, "Response lacks paragraph structure", 0.05
        
        return True, "Coherence check passed", 0.0
    
    def _enhance_response(
        self,
        response: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Enhance response with improvements"""
        
        enhanced = response
        
        # Add disclaimer if missing and needed
        if not re.search(r'konsultasi.*?ahli hukum', enhanced, re.IGNORECASE):
            if any(word in enhanced.lower() for word in ['sanksi', 'pidana', 'denda', 'gugatan']):
                enhanced += "\n\n**Catatan**: Untuk kasus atau situasi spesifik, disarankan untuk berkonsultasi dengan ahli hukum atau pengacara profesional."
        
        # Add reference section if citations exist but no reference list
        if '[Dokumen' in enhanced and 'REFERENSI' not in enhanced.upper():
            from .citation_formatter import CitationFormatter
            formatter = CitationFormatter(self.config)
            references = formatter.format_reference_list(results, max_items=5)
            enhanced += f"\n\n{references}"
        
        return enhanced
    
    def check_factual_consistency(
        self,
        response: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if response is consistent with retrieved documents
        
        Args:
            response: Generated response
            results: Retrieved documents
            
        Returns:
            Consistency check result
        """
        self.logger.info("Checking factual consistency")
        
        consistency_result = {
            'is_consistent': True,
            'inconsistencies': [],
            'confidence': 1.0
        }
        
        # Extract claims from response
        claims = self._extract_claims(response)
        
        # Check each claim against documents
        for claim in claims:
            # Simple keyword matching for now
            # In production, use more sophisticated fact-checking
            claim_lower = claim.lower()
            
            found_support = False
            for result in results:
                content = result.get('record', {}).get('content', '').lower()
                
                # Check if claim keywords appear in content
                claim_words = set(re.findall(r'\w+', claim_lower))
                content_words = set(re.findall(r'\w+', content))
                
                overlap = len(claim_words & content_words)
                if overlap > len(claim_words) * 0.5:  # 50% overlap
                    found_support = True
                    break
            
            if not found_support and len(claim_words) > 5:
                consistency_result['inconsistencies'].append(claim)
                consistency_result['confidence'] *= 0.9
        
        if consistency_result['inconsistencies']:
            consistency_result['is_consistent'] = False
            self.logger.warning("Potential inconsistencies detected", {
                "count": len(consistency_result['inconsistencies'])
            })
        
        return consistency_result
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response"""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter out very short sentences and questions
            if len(sentence) > 30 and '?' not in sentence:
                claims.append(sentence)
        
        return claims
    
    def sanitize_response(self, response: str) -> str:
        """
        Sanitize response by removing sensitive or inappropriate content
        
        Args:
            response: Original response
            
        Returns:
            Sanitized response
        """
        sanitized = response
        
        # Remove model artifacts
        artifacts = [
            r'<think>.*?</think>',
            r'<\|.*?\|>',
            r'\[INST\].*?\[/INST\]'
        ]
        
        for pattern in artifacts:
            sanitized = re.sub(pattern, '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\n\n\n+', '\n\n', sanitized)
        sanitized = re.sub(r'  +', ' ', sanitized)
        
        return sanitized.strip()
    
    def get_quality_report(
        self,
        validation_result: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable quality report
        
        Args:
            validation_result: Validation result dictionary
            
        Returns:
            Formatted quality report
        """
        report = ["LAPORAN KUALITAS RESPONS", "=" * 50, ""]
        
        # Overall status
        status = "✅ VALID" if validation_result['is_valid'] else "❌ TIDAK VALID"
        report.append(f"Status: {status}")
        report.append(f"Skor Kualitas: {validation_result['quality_score']:.2%}")
        report.append("")
        
        # Checks passed
        if validation_result['checks_passed']:
            report.append("Pemeriksaan yang Lulus:")
            for check in validation_result['checks_passed']:
                report.append(f"  ✓ {check}")
            report.append("")
        
        # Checks failed
        if validation_result['checks_failed']:
            report.append("Pemeriksaan yang Gagal:")
            for check in validation_result['checks_failed']:
                report.append(f"  ✗ {check}")
            report.append("")
        
        # Errors
        if validation_result['errors']:
            report.append("Error:")
            for error in validation_result['errors']:
                report.append(f"  • {error}")
            report.append("")
        
        # Warnings
        if validation_result['warnings']:
            report.append("Peringatan:")
            for warning in validation_result['warnings']:
                report.append(f"  ⚠ {warning}")
            report.append("")
        
        return "\n".join(report)