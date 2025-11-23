"""
LLM Engine for Indonesian Legal RAG System
Handles model loading, inference, and generation with proper error handling

File: core/generation/llm_engine.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional, Generator
import time
from logger_utils import get_logger
from config import (
    LLM_MODEL,
    MAX_LENGTH,
    CACHE_DIR,
    DEVICE,
    get_model_path,
    USE_LOCAL_MODELS
)


class LLMEngine:
    """
    LLM Engine for generating legal responses
    Supports both synchronous and streaming generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("LLMEngine")
        self.config = config
        
        # Model configuration - use get_model_path for local model support
        self.model_name = config.get('llm_model', get_model_path('llm'))
        self.max_length = config.get('max_length', MAX_LENGTH)
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Generation parameters from config
        self.temperature = config.get('temperature', 0.7)
        self.max_new_tokens = config.get('max_new_tokens', 2048)
        self.top_p = config.get('top_p', 1.0)
        self.top_k = config.get('top_k', 20)
        self.min_p = config.get('min_p', 0.1)
        self.repetition_penalty = config.get('repetition_penalty', 1.1)
        
        # Model and tokenizer (lazy loaded)
        self._model = None
        self._tokenizer = None
        
        self.logger.info("LLMEngine initialized", {
            "model": self.model_name,
            "device": str(self.device),
            "max_new_tokens": self.max_new_tokens
        })
    
    def load_model(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Load LLM model with retry logic
        
        Args:
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if self._model is not None:
            self.logger.debug("Model already loaded")
            return True
        
        self.logger.info("Loading LLM model", {"model": self.model_name})
        
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt}/{max_retries}")
                
                # Load tokenizer
                self.logger.debug("Loading tokenizer")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=CACHE_DIR,
                    trust_remote_code=True
                )
                
                # Set pad token if not exists
                # IMPORTANT: Don't use eos_token as pad_token - it causes generation to stop immediately
                need_resize_embeddings = False
                if self._tokenizer.pad_token is None:
                    # Try to use unk_token first, but verify it's different from eos
                    if (self._tokenizer.unk_token is not None and
                        self._tokenizer.unk_token_id != self._tokenizer.eos_token_id):
                        self._tokenizer.pad_token = self._tokenizer.unk_token
                        self.logger.debug("Set pad_token to unk_token")
                    else:
                        # Add a dedicated pad token - safest option
                        self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        need_resize_embeddings = True
                        self.logger.debug("Added new [PAD] token")

                # Extra safety check: if pad_token_id equals eos_token_id, force add new token
                if self._tokenizer.pad_token_id == self._tokenizer.eos_token_id:
                    self.logger.warning(f"pad_token_id ({self._tokenizer.pad_token_id}) equals eos_token_id, adding dedicated [PAD] token")
                    self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    need_resize_embeddings = True

                # Set padding side to left for decoder-only models
                self._tokenizer.padding_side = 'left'
                self.logger.debug(f"Set padding_side to left, pad_token_id={self._tokenizer.pad_token_id}, eos_token_id={self._tokenizer.eos_token_id}")

                # Load model
                self.logger.debug("Loading model weights")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=CACHE_DIR,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    trust_remote_code=True,
                    device_map='auto' if self.device.type == 'cuda' else None
                )
                
                if self.device.type != 'cuda':
                    self._model.to(self.device)

                # Resize embeddings if we added new tokens
                if need_resize_embeddings:
                    self._model.resize_token_embeddings(len(self._tokenizer))
                    self.logger.debug(f"Resized token embeddings to {len(self._tokenizer)}")

                self._model.eval()
                
                self.logger.success("LLM model loaded successfully", {
                    "model": self.model_name,
                    "device": str(self.device),
                    "dtype": str(self._model.dtype)
                })
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load model (attempt {attempt})", {
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                if attempt < max_retries:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Max retries reached, giving up")
                    return False
        
        return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate response from prompt (synchronous)
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Optional stop sequences
            
        Returns:
            Dictionary with generated text and metadata
        """
        if self._model is None:
            self.logger.error("Model not loaded, call load_model() first")
            return {
                'generated_text': '',
                'error': 'Model not loaded',
                'success': False
            }
        
        self.logger.info("Starting generation", {
            "prompt_length": len(prompt),
            "max_new_tokens": max_new_tokens or self.max_new_tokens
        })
        
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            self.logger.debug("Input tokenized", {
                "input_tokens": input_length
            })
            
            # Generation parameters
            gen_kwargs = {
                'max_new_tokens': max_new_tokens or self.max_new_tokens,
                'temperature': temperature or self.temperature,
                'top_p': top_p or self.top_p,
                'top_k': top_k or self.top_k,
                'repetition_penalty': self.repetition_penalty,
                'do_sample': True,
                'pad_token_id': self._tokenizer.pad_token_id,
                'eos_token_id': self._tokenizer.eos_token_id,
            }
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Decode
            generated_ids = outputs[0][input_length:]
            generated_text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            # Debug logging for short generations
            tokens_generated = len(generated_ids)
            if tokens_generated <= 5:
                self.logger.warning(f"Very short generation detected: {tokens_generated} tokens")
                raw_tokens = [self._tokenizer.decode([tid], skip_special_tokens=False) for tid in generated_ids.tolist()]
                self.logger.debug(f"Generated token IDs: {generated_ids.tolist()}")
                self.logger.debug(f"Generated tokens (raw): {raw_tokens}")
                self.logger.debug(f"EOS token ID: {self._tokenizer.eos_token_id}, PAD token ID: {self._tokenizer.pad_token_id}")
                if len(generated_ids) > 0 and generated_ids[0].item() == self._tokenizer.eos_token_id:
                    self.logger.error("First generated token is EOS - model may have wrong config or prompt issue")

            # Post-process
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text[:generated_text.index(stop_seq)]
            
            generation_time = time.time() - start_time
            tokens_generated = len(generated_ids)
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            self.logger.success("Generation completed", {
                "tokens_generated": tokens_generated,
                "generation_time": f"{generation_time:.2f}s",
                "tokens_per_second": f"{tokens_per_second:.1f}"
            })
            
            return {
                'generated_text': generated_text.strip(),
                'tokens_generated': tokens_generated,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            self.logger.error("Generation failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()[:500]
            })
            
            return {
                'generated_text': '',
                'error': str(e),
                'success': False
            }
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate response with streaming (yields tokens as generated)
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Optional stop sequences
            
        Yields:
            Dictionary with token and metadata
        """
        if self._model is None:
            self.logger.error("Model not loaded, call load_model() first")
            yield {
                'token': '',
                'error': 'Model not loaded',
                'done': True,
                'success': False
            }
            return
        
        self.logger.info("Starting streaming generation")
        
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # Generation parameters
            gen_kwargs = {
                'max_new_tokens': max_new_tokens or self.max_new_tokens,
                'temperature': temperature or self.temperature,
                'top_p': top_p or self.top_p,
                'top_k': top_k or self.top_k,
                'repetition_penalty': self.repetition_penalty,
                'do_sample': True,
                'pad_token_id': self._tokenizer.pad_token_id,
                'eos_token_id': self._tokenizer.eos_token_id,
            }
            
            # Stream generation
            tokens_generated = 0
            full_text = ""
            
            with torch.no_grad():
                # Use model.generate with return_dict_in_generate for streaming
                current_ids = inputs['input_ids']
                
                for _ in range(gen_kwargs['max_new_tokens']):
                    outputs = self._model(
                        input_ids=current_ids,
                        attention_mask=inputs['attention_mask']
                    )
                    
                    # Get next token logits
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / gen_kwargs['temperature']
                    
                    # Apply top-k and top-p filtering
                    filtered_logits = self._top_k_top_p_filtering(
                        next_token_logits,
                        top_k=gen_kwargs['top_k'],
                        top_p=gen_kwargs['top_p']
                    )
                    
                    # Sample next token
                    probs = torch.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Check for EOS
                    if next_token.item() == self._tokenizer.eos_token_id:
                        break
                    
                    # Decode token
                    token_text = self._tokenizer.decode(next_token[0], skip_special_tokens=True)
                    full_text += token_text
                    tokens_generated += 1
                    
                    # Check stop sequences
                    if stop_sequences:
                        should_stop = False
                        for stop_seq in stop_sequences:
                            if stop_seq in full_text:
                                should_stop = True
                                break
                        if should_stop:
                            break
                    
                    # Yield token
                    yield {
                        'token': token_text,
                        'tokens_generated': tokens_generated,
                        'done': False,
                        'success': True
                    }
                    
                    # Append to current_ids
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    
                    # Update attention mask
                    inputs['attention_mask'] = torch.cat([
                        inputs['attention_mask'],
                        torch.ones((1, 1), device=self.device, dtype=torch.long)
                    ], dim=1)
            
            generation_time = time.time() - start_time
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            self.logger.success("Streaming generation completed", {
                "tokens_generated": tokens_generated,
                "generation_time": f"{generation_time:.2f}s",
                "tokens_per_second": f"{tokens_per_second:.1f}"
            })
            
            # Final yield
            yield {
                'token': '',
                'tokens_generated': tokens_generated,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'done': True,
                'success': True
            }
            
        except Exception as e:
            self.logger.error("Streaming generation failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            yield {
                'token': '',
                'error': str(e),
                'done': True,
                'success': False
            }
    
    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:
        """Apply top-k and top-p filtering to logits"""
        
        if top_k > 0:
            # Remove tokens with rank < top_k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1,
                index=sorted_indices,
                src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        
        return logits
    
    def unload_model(self):
        """Unload model to free memory"""
        self.logger.info("Unloading LLM model")
        
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.success("LLM model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_loaded': self._model is not None,
            'model_name': self.model_name,
            'device': str(self.device),
            'max_length': self.max_length,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k
        }


# Global LLM engine instance
_llm_engine = None


def get_llm_engine(config: Optional[Dict[str, Any]] = None) -> LLMEngine:
    """
    Get global LLM engine instance (singleton)
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LLMEngine instance
    """
    global _llm_engine
    
    if _llm_engine is None:
        from config import get_default_config
        cfg = config or get_default_config()
        _llm_engine = LLMEngine(cfg)
    
    return _llm_engine