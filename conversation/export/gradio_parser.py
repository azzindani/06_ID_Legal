"""
Gradio Content Parser - Parse formatted chat content for export

Extracts structured data from HTML-formatted chat responses.
Used by API-based UI to prepare data for standard exporters.

File: conversation/export/gradio_parser.py
"""

import re
from typing import Dict, Any, List, Optional


def extract_text_content(content: Any) -> str:
    """
    Extract plain text from various Gradio message formats.
    Handles: str, list of dicts [{'text': '...'}], dict {'text': '...'}, etc.
    """
    if content is None:
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, dict):
        # Format: {'text': '...', 'type': 'text'}
        return str(content.get('text', content.get('content', str(content))))
    
    if isinstance(content, (list, tuple)):
        # Format: [{'text': '...', 'type': 'text'}, ...]
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get('text', item.get('content', ''))))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return '\n'.join(parts)
    
    return str(content)


def parse_gradio_content(raw_content: Any) -> Dict[str, str]:
    """
    Parse the raw chat content to extract structured sections.
    The content contains HTML details/summary tags for thinking, sources, research process.
    
    Args:
        raw_content: Raw content from chat response (may be string, dict, or list)
        
    Returns:
        Dict with 'thinking', 'answer', 'sources', 'research_process' keys
    """
    result = {
        'thinking': '',
        'answer': '',
        'sources': '',
        'research_process': ''
    }
    
    if not raw_content:
        return result
    
    # First extract plain text from Gradio format
    content = extract_text_content(raw_content)
    
    if not content:
        return result
    
    # Extract thinking from <details><summary>🧠 ... </summary>...</details>
    thinking_match = re.search(
        r'<details>\s*<summary>🧠.*?Proses Berpikir.*?</summary>\s*(.*?)\s*</details>',
        content, re.DOTALL | re.IGNORECASE
    )
    if thinking_match:
        result['thinking'] = thinking_match.group(1).strip()
    
    # Extract answer - after "### ✅ Jawaban" or just the main text
    answer_match = re.search(
        r'###\s*✅\s*Jawaban\s*\n+(.*?)(?=\n---|\n<details|$)',
        content, re.DOTALL
    )
    if answer_match:
        result['answer'] = answer_match.group(1).strip()
    elif '### ✅ Jawaban' not in content:
        # If no clear answer header, try to get text between thinking and sources
        clean = re.sub(r'<details>.*?</details>', '', content, flags=re.DOTALL)
        clean = re.sub(r'<[^>]+>', '', clean)  # Remove any remaining HTML
        result['answer'] = clean.strip()
    
    # Extract sources from <details><summary>📖 ... Sumber Hukum...</summary>...</details>
    sources_match = re.search(
        r'<details>\s*<summary>📖.*?Sumber Hukum.*?</summary>\s*(.*?)\s*</details>',
        content, re.DOTALL | re.IGNORECASE
    )
    if sources_match:
        result['sources'] = sources_match.group(1).strip()
    
    # Extract research process from <details><summary>🔬 ... Detail...</summary>...</details>
    research_match = re.search(
        r'<details>\s*<summary>🔬.*?Detail.*?</summary>\s*(.*?)\s*</details>',
        content, re.DOTALL | re.IGNORECASE
    )
    if research_match:
        result['research_process'] = research_match.group(1).strip()
    
    # Clean HTML tags from all fields
    for key in result:
        if result[key]:
            result[key] = re.sub(r'<[^>]+>', '', result[key])
            result[key] = result[key].strip()
    
    return result


def history_to_session_data(
    history: List[Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert Gradio chat history to session data format for exporters.
    
    Args:
        history: Gradio chatbot history (list of dicts or tuples)
        session_id: Optional session ID
        
    Returns:
        Session data dict compatible with standard exporters
    """
    from datetime import datetime
    
    session_data = {
        'id': session_id or 'export',
        'session_id': session_id or 'export',
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'turns': [],
        'metadata': {
            'total_queries': 0,
            'source': 'api_ui'
        }
    }
    
    i = 0
    turn_num = 1
    while i < len(history):
        item = history[i]
        
        if isinstance(item, dict) and item.get('role') == 'user' and i + 1 < len(history):
            # Extract user content
            user_raw = item.get('content', '')
            user_content = extract_text_content(user_raw)
            
            # Extract assistant content
            next_item = history[i + 1]
            if isinstance(next_item, dict):
                assistant_raw = next_item.get('content', '')
            else:
                assistant_raw = next_item
            assistant_content = extract_text_content(assistant_raw)
            
            # Parse the assistant content for structured data
            parsed = parse_gradio_content(assistant_content)
            
            # Build citations from sources if available
            citations = []
            if parsed['sources']:
                # Extract regulation references from sources text
                reg_matches = re.findall(
                    r'(\w+(?:\s+\w+)*)\s+No\.?\s*(\d+)\s*/?\s*(?:Tahun\s+)?(\d{4})',
                    parsed['sources']
                )
                for reg_type, reg_num, year in reg_matches:
                    citations.append({
                        'regulation_type': reg_type,
                        'regulation_number': reg_num,
                        'year': year,
                        'about': ''
                    })
            
            session_data['turns'].append({
                'turn_number': turn_num,
                'query': user_content,
                'answer': parsed['answer'] or assistant_content,
                'thinking': parsed['thinking'],
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'citations': citations,
                    'research_log': {'details': parsed['research_process']} if parsed['research_process'] else {}
                }
            })
            
            turn_num += 1
            i += 2
        else:
            i += 1
    
    session_data['metadata']['total_queries'] = len(session_data['turns'])
    
    return session_data
