"""
LLM Engine for Indonesian Legal RAG System
Handles model loading, inference, and generation with proper error handling

File: core/generation/llm_engine.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Dict, Any, List, Optional, Generator
import time
from threading import Thread
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

        # CRITICAL: Clear GPU cache BEFORE generation to prevent OOM
        # Especially important for long prompts (thinking modes)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.debug("Cleared GPU cache before generation")

        start_time = time.time()

        try:
            # Apply chat template if available (required for instruction-tuned models)
            formatted_prompt = prompt
            if hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template:
                try:
                    # Structure as chat messages
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    self.logger.debug("Applied chat template to prompt")
                except Exception as e:
                    self.logger.warning(f"Failed to apply chat template: {e}, using raw prompt")
                    formatted_prompt = prompt

            # Tokenize input
            inputs = self._tokenizer(
                formatted_prompt,
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
                # use_cache=True by default - cache is used WITHIN generation, not between calls
                # Deleting outputs frees the cache (past_key_values)
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

            # CRITICAL: Clean up tensors to prevent OOM on next generation
            # Delete inputs and outputs to free GPU memory immediately
            del inputs
            del outputs
            del generated_ids
            # Force garbage collection to free memory NOW, not later
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure cleanup completes
                self.logger.debug("Cleaned up generation tensors and cleared CUDA cache")

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
        Generate response with streaming using TextIteratorStreamer

        This uses threading for efficient parallel token generation,
        yielding tokens as they are produced by the model.

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

        self.logger.info("Starting streaming generation with TextIteratorStreamer")

        # CRITICAL: Clear GPU cache BEFORE generation to prevent OOM
        # Especially important for long prompts (thinking modes)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.debug("Cleared GPU cache before streaming generation")

        start_time = time.time()

        try:
            # Apply chat template if available (required for instruction-tuned models)
            formatted_prompt = prompt
            if hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    self.logger.debug("Applied chat template to prompt (stream)")
                except Exception as e:
                    self.logger.warning(f"Failed to apply chat template: {e}, using raw prompt")
                    formatted_prompt = prompt

            # Tokenize input
            inputs = self._tokenizer(
                formatted_prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            # Create TextIteratorStreamer for efficient streaming
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Generation parameters
            gen_kwargs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'streamer': streamer,
                'max_new_tokens': max_new_tokens or self.max_new_tokens,
                'temperature': temperature or self.temperature,
                'top_p': top_p or self.top_p,
                'top_k': top_k or self.top_k,
                'repetition_penalty': self.repetition_penalty,
                'do_sample': True,
                # use_cache=True by default - cache is used WITHIN generation, not between calls
                # The thread-based generation completes and cache is freed when outputs are deleted
                'pad_token_id': self._tokenizer.pad_token_id,
                'eos_token_id': self._tokenizer.eos_token_id,
            }

            # Start generation in background thread
            thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
            thread.start()

            # Stream tokens as they are generated
            tokens_generated = 0
            full_text = ""

            for new_text in streamer:
                full_text += new_text
                tokens_generated += 1

                # Check stop sequences
                should_stop = False
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in full_text:
                            should_stop = True
                            break

                if should_stop:
                    break

                # Yield token
                yield {
                    'token': new_text,
                    'tokens_generated': tokens_generated,
                    'done': False,
                    'success': True
                }

            # Wait for generation thread to complete
            thread.join()

            generation_time = time.time() - start_time
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

            self.logger.success("Streaming generation completed", {
                "tokens_generated": tokens_generated,
                "generation_time": f"{generation_time:.2f}s",
                "tokens_per_second": f"{tokens_per_second:.1f}"
            })

            # CRITICAL: Clean up tensors to prevent OOM on next generation
            # The inputs tensor and KV cache from generation stay in GPU memory
            # until explicitly deleted and cache is cleared
            del inputs
            del streamer
            # Force garbage collection to free memory NOW, not later
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure cleanup completes
                self.logger.debug("Cleaned up generation tensors and cleared CUDA cache")

            # Final yield
            yield {
                'token': '',
                'tokens_generated': tokens_generated,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'full_text': full_text,
                'done': True,
                'success': True
            }

        except Exception as e:
            self.logger.error("Streaming generation failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })

            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()[:500]
            })

            yield {
                'token': '',
                'error': str(e),
                'done': True,
                'success': False
            }

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