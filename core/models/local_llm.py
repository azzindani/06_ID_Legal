# core/models/local_llm.py
"""
Local HuggingFace model wrapper with GPU support.
"""
import torch
import time
from typing import Dict, Any, Optional, Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from core.models.base_llm import BaseLLM, LLMType, GenerationConfig, LLMResponse
from utils.logging_config import get_logger, log_performance

logger = get_logger(__name__)

class LocalLLM(BaseLLM):
    """
    Wrapper for local HuggingFace models.
    Supports streaming, GPU acceleration, and comprehensive error handling.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        device: str = 'cuda',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize local LLM.
        
        Args:
            model_name: HuggingFace model identifier
            config: Model configuration
            device: Device for inference ('cuda', 'cpu', 'mps')
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
        """
        super().__init__(model_name, config)
        
        self.llm_type = LLMType.LOCAL
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        logger.info(f"LocalLLM configured: device={device}, 8bit={load_in_8bit}, 4bit={load_in_4bit}")
    
    def load(self):
        """Load model and tokenizer."""
        if self._loaded:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading local model: {self.model_name}")
        
        try:
            # Load tokenizer
            logger.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with appropriate precision
            logger.debug("Loading model...")
            
            load_kwargs = {
                'device_map': 'auto',
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32
            }
            
            if self.load_in_8bit:
                load_kwargs['load_in_8bit'] = True
                logger.info("Loading in 8-bit mode")
            elif self.load_in_4bit:
                load_kwargs['load_in_4bit'] = True
                logger.info("Loading in 4-bit mode")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            self.model.eval()
            self._loaded = True
            
            logger.info(f"✅ Model loaded successfully: {self.model_name}")
            
            # Log model info
            if hasattr(self.model, 'num_parameters'):
                num_params = self.model.num_parameters() / 1e9
                logger.info(f"Model size: {num_params:.2f}B parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def unload(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")
    
    @log_performance(logger)
    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text synchronously.
        
        Args:
            prompt: Input prompt
            generation_config: Generation parameters
            **kwargs: Additional generation arguments
            
        Returns:
            LLMResponse with generated text
        """
        if not self._loaded:
            self.load()
        
        if generation_config is None:
            generation_config = GenerationConfig()
        
        logger.info(f"Generating (sync): max_tokens={generation_config.max_new_tokens}, temp={generation_config.temperature}")
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            input_length = inputs['input_ids'].shape[1]
            
            logger.debug(f"Input tokens: {input_length}")
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config.max_new_tokens,
                    temperature=generation_config.temperature if generation_config.do_sample else 1.0,
                    top_p=generation_config.top_p,
                    top_k=generation_config.top_k,
                    do_sample=generation_config.do_sample,
                    repetition_penalty=generation_config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            # Calculate tokens
            output_length = outputs.shape[1] - input_length
            total_tokens = input_length + output_length
            
            # Update stats
            elapsed = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['total_tokens_used'] += total_tokens
            self.stats['total_generation_time'] += elapsed
            
            logger.info(
                f"Generation complete: {output_length} tokens in {elapsed:.2f}s "
                f"({output_length/elapsed:.1f} tokens/s)"
            )
            
            return LLMResponse(
                text=generated_text,
                finish_reason='stop',
                tokens_used=total_tokens,
                model=self.model_name,
                metadata={
                    'input_tokens': input_length,
                    'output_tokens': output_length,
                    'generation_time': elapsed
                }
            )
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Generation failed: {e}", exc_info=True)
            
            return LLMResponse(
                text="",
                finish_reason='error',
                model=self.model_name,
                metadata={'error': str(e)}
            )
    
    def generate_stream(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming.
        
        Args:
            prompt: Input prompt
            generation_config: Generation parameters
            **kwargs: Additional generation arguments
            
        Yields:
            Text chunks as they're generated
        """
        if not self._loaded:
            self.load()
        
        if generation_config is None:
            generation_config = GenerationConfig()
        
        logger.info(f"Generating (stream): max_tokens={generation_config.max_new_tokens}")
        
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            input_length = inputs['input_ids'].shape[1]
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Prepare generation kwargs
            generation_kwargs = {
                **inputs,
                'streamer': streamer,
                'max_new_tokens': generation_config.max_new_tokens,
                'temperature': generation_config.temperature if generation_config.do_sample else 1.0,
                'top_p': generation_config.top_p,
                'top_k': generation_config.top_k,
                'do_sample': generation_config.do_sample,
                'repetition_penalty': generation_config.repetition_penalty,
                'pad_token_id': self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Start generation in background thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens
            token_count = 0
            for new_text in streamer:
                token_count += 1
                yield new_text
            
            thread.join()
            
            # Update stats
            elapsed = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['total_tokens_used'] += (input_length + token_count)
            self.stats['total_generation_time'] += elapsed
            
            logger.info(
                f"Streaming complete: {token_count} tokens in {elapsed:.2f}s"
            )
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            yield f"[Error: {str(e)}]"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self._loaded:
            self.load()
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def is_available(self) -> bool:
        """Check if model is loaded and working."""
        if not self._loaded:
            return False
        
        try:
            # Quick test generation
            test_input = self.tokenizer("test", return_tensors='pt').to(self.device)
            with torch.no_grad():
                _ = self.model.generate(**test_input, max_new_tokens=1)
            return True
        except Exception as e:
            logger.error(f"Availability check failed: {e}")
            return False