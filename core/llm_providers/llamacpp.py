"""
LlamaCpp Provider - GGUF Model Inference via llama-cpp-python

Provides local GGUF model inference with hybrid CPU/GPU support.
Follows valve architecture for hot-swapping with other providers.

File: core/llm_providers/llamacpp.py
"""

import time
from typing import Dict, Any, List, Optional, Generator
from pathlib import Path
from .base import LLMProviderBase

# Import logger
try:
    from utils.logger_utils import get_logger
    logger = get_logger("LlamaCppProvider")
except ImportError:
    import logging
    logger = logging.getLogger("LlamaCppProvider")


class LlamaCppProvider(LLMProviderBase):
    """
    LlamaCpp provider for GGUF model inference.
    
    Uses llama-cpp-python for CPU/GPU inference with:
    - Hybrid CPU/GPU layer offloading
    - 32K context support
    - Streaming generation
    
    Default model: Azzindani/Deepseek_ID_Legal_Preview_GGUF
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        n_ctx: int = 32768,
        n_gpu_layers: int = -1,
        n_threads: int = 0,
        n_threads_batch: int = 0,
        use_mmap: bool = True,
        use_mlock: bool = False,
        offload_kqv: bool = True,
        flash_attn: bool = False,
        main_gpu: int = 0,
        split_mode: str = "layer",
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize LlamaCpp provider.
        
        Args:
            model_path: Path to local GGUF file (optional if repo_id provided)
            repo_id: HuggingFace repo ID for auto-download
            filename: GGUF filename in the repo
            n_ctx: Context window size (default 32K)
            n_gpu_layers: GPU layers (-1=all, 0=CPU only, N=hybrid)
            n_threads: CPU threads (0=auto)
            n_threads_batch: Batch processing threads
            use_mmap: Memory-map model file
            use_mlock: Lock model in RAM
            offload_kqv: Offload KV cache to GPU
            flash_attn: Use flash attention
            main_gpu: Primary GPU index
            split_mode: GPU split mode (layer, row, none)
            verbose: Enable verbose logging
        """
        # Import config for defaults
        try:
            from config import (
                LLAMACPP_REPO_ID, LLAMACPP_FILENAME,
                LLAMACPP_N_CTX, LLAMACPP_N_GPU_LAYERS,
                LLAMACPP_N_THREADS, LLAMACPP_N_THREADS_BATCH,
                LLAMACPP_USE_MMAP, LLAMACPP_USE_MLOCK,
                LLAMACPP_OFFLOAD_KQV, LLAMACPP_FLASH_ATTN,
                LLAMACPP_MAIN_GPU, LLAMACPP_SPLIT_MODE
            )
            default_repo = LLAMACPP_REPO_ID
            default_filename = LLAMACPP_FILENAME
            default_n_ctx = LLAMACPP_N_CTX
            default_n_gpu_layers = LLAMACPP_N_GPU_LAYERS
            default_n_threads = LLAMACPP_N_THREADS
            default_n_threads_batch = LLAMACPP_N_THREADS_BATCH
            default_use_mmap = LLAMACPP_USE_MMAP
            default_use_mlock = LLAMACPP_USE_MLOCK
            default_offload_kqv = LLAMACPP_OFFLOAD_KQV
            default_flash_attn = LLAMACPP_FLASH_ATTN
            default_main_gpu = LLAMACPP_MAIN_GPU
            default_split_mode = LLAMACPP_SPLIT_MODE
        except ImportError:
            default_repo = "Azzindani/Deepseek_ID_Legal_Preview_GGUF"
            default_filename = "ID_Legal_Assistant_Q4_K_M.gguf"
            default_n_ctx = 32768
            default_n_gpu_layers = -1
            default_n_threads = 0
            default_n_threads_batch = 0
            default_use_mmap = True
            default_use_mlock = False
            default_offload_kqv = True
            default_flash_attn = False
            default_main_gpu = 0
            default_split_mode = "layer"
        
        # Store config with defaults
        self._model_path = model_path
        self._repo_id = repo_id or default_repo
        self._filename = filename or default_filename
        self._n_ctx = n_ctx if n_ctx != 32768 else default_n_ctx
        self._n_gpu_layers = n_gpu_layers if n_gpu_layers != -1 else default_n_gpu_layers
        self._n_threads = n_threads if n_threads != 0 else default_n_threads
        self._n_threads_batch = n_threads_batch if n_threads_batch != 0 else default_n_threads_batch
        self._use_mmap = use_mmap if use_mmap != True else default_use_mmap
        self._use_mlock = use_mlock if use_mlock != False else default_use_mlock
        self._offload_kqv = offload_kqv if offload_kqv != True else default_offload_kqv
        self._flash_attn = flash_attn if flash_attn != False else default_flash_attn
        self._main_gpu = main_gpu if main_gpu != 0 else default_main_gpu
        self._split_mode = split_mode if split_mode != "layer" else default_split_mode
        self._verbose = verbose
        
        self._llm = None
        self._model_loaded = False
        self._actual_model_path = None
        
        logger.info(f"LlamaCpp provider initialized (model not loaded yet)")
        logger.info(f"Config: n_ctx={self._n_ctx}, n_gpu_layers={self._n_gpu_layers}")
    
    @property
    def provider_name(self) -> str:
        return "llamacpp"
    
    @property
    def model_name(self) -> str:
        if self._actual_model_path:
            return Path(self._actual_model_path).stem
        return self._filename.replace('.gguf', '') if self._filename else 'unknown'
    
    def _download_model(self) -> str:
        """Download model from HuggingFace if needed."""
        from huggingface_hub import hf_hub_download
        
        logger.info(f"Downloading model from {self._repo_id}/{self._filename}...")
        
        model_path = hf_hub_download(
            repo_id=self._repo_id,
            filename=self._filename,
            resume_download=True
        )
        
        logger.info(f"Model downloaded to: {model_path}")
        return model_path
    
    def load_model(self) -> bool:
        """
        Load the GGUF model.
        
        Returns:
            True if successful, False otherwise
        """
        if self._model_loaded and self._llm is not None:
            logger.debug("Model already loaded")
            return True
        
        try:
            # Import llama-cpp-python
            from llama_cpp import Llama
            
            # Determine model path
            if self._model_path and Path(self._model_path).exists():
                self._actual_model_path = self._model_path
            else:
                self._actual_model_path = self._download_model()
            
            logger.info(f"Loading GGUF model: {self._actual_model_path}")
            logger.info(f"GPU layers: {self._n_gpu_layers}, Context: {self._n_ctx}")
            
            # Build kwargs for Llama
            llama_kwargs = {
                'model_path': self._actual_model_path,
                'n_ctx': self._n_ctx,
                'n_gpu_layers': self._n_gpu_layers,
                'use_mmap': self._use_mmap,
                'use_mlock': self._use_mlock,
                'verbose': self._verbose,
            }
            
            # Add optional params
            if self._n_threads > 0:
                llama_kwargs['n_threads'] = self._n_threads
            if self._n_threads_batch > 0:
                llama_kwargs['n_threads_batch'] = self._n_threads_batch
            if self._flash_attn:
                llama_kwargs['flash_attn'] = True
            if self._offload_kqv:
                llama_kwargs['offload_kqv'] = True
            if self._main_gpu != 0:
                llama_kwargs['main_gpu'] = self._main_gpu
            
            # Load model
            start_time = time.time()
            self._llm = Llama(**llama_kwargs)
            load_time = time.time() - start_time
            
            self._model_loaded = True
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            return True
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self):
        """Unload model to free memory."""
        if self._llm:
            del self._llm
            self._llm = None
            self._model_loaded = False
            logger.info("LlamaCpp model unloaded")
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response synchronously.
        """
        if not self._model_loaded or self._llm is None:
            return {
                'generated_text': '',
                'success': False,
                'error': 'Model not loaded. Call load_model() first.',
                'provider': self.provider_name,
                'model': self.model_name,
            }
        
        if not self.validate_prompt(prompt):
            return {
                'generated_text': '',
                'success': False,
                'error': 'Invalid or empty prompt',
                'provider': self.provider_name,
                'model': self.model_name,
            }
        
        try:
            start_time = time.time()
            
            # Build generation params
            gen_kwargs = {
                'max_tokens': max_new_tokens or 2048,
                'echo': False,
            }
            
            if temperature is not None:
                gen_kwargs['temperature'] = temperature
            if top_p is not None:
                gen_kwargs['top_p'] = top_p
            if top_k is not None:
                gen_kwargs['top_k'] = top_k
            if stop_sequences:
                gen_kwargs['stop'] = stop_sequences
            
            # Generate
            output = self._llm(prompt, **gen_kwargs)
            
            elapsed = time.time() - start_time
            
            # Extract text
            generated_text = output['choices'][0]['text'] if output.get('choices') else ''
            tokens_generated = output.get('usage', {}).get('completion_tokens', 0)
            prompt_tokens = output.get('usage', {}).get('prompt_tokens', 0)
            
            return {
                'generated_text': generated_text,
                'success': True,
                'error': None,
                'tokens_generated': tokens_generated,
                'prompt_tokens': prompt_tokens,
                'total_tokens': prompt_tokens + tokens_generated,
                'generation_time': elapsed,
                'tokens_per_second': tokens_generated / elapsed if elapsed > 0 else 0,
                'cost_usd': 0.0,  # Local = free
                'provider': self.provider_name,
                'model': self.model_name,
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                'generated_text': '',
                'success': False,
                'error': str(e),
                'provider': self.provider_name,
                'model': self.model_name,
            }
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response token by token.
        """
        if not self._model_loaded or self._llm is None:
            yield {
                'token': '',
                'done': True,
                'success': False,
                'error': 'Model not loaded. Call load_model() first.',
            }
            return
        
        if not self.validate_prompt(prompt):
            yield {
                'token': '',
                'done': True,
                'success': False,
                'error': 'Invalid or empty prompt',
            }
            return
        
        try:
            start_time = time.time()
            full_text = ""
            tokens_generated = 0
            
            # Build generation params
            gen_kwargs = {
                'max_tokens': max_new_tokens or 2048,
                'echo': False,
                'stream': True,
            }
            
            if temperature is not None:
                gen_kwargs['temperature'] = temperature
            if top_p is not None:
                gen_kwargs['top_p'] = top_p
            if top_k is not None:
                gen_kwargs['top_k'] = top_k
            if stop_sequences:
                gen_kwargs['stop'] = stop_sequences
            
            # Stream generation
            for output in self._llm(prompt, **gen_kwargs):
                token = output['choices'][0]['text'] if output.get('choices') else ''
                full_text += token
                tokens_generated += 1
                
                elapsed = time.time() - start_time
                
                yield {
                    'token': token,
                    'done': False,
                    'success': True,
                    'error': None,
                    'tokens_generated': tokens_generated,
                    'generation_time': elapsed,
                    'tokens_per_second': tokens_generated / elapsed if elapsed > 0 else 0,
                }
            
            # Final chunk
            elapsed = time.time() - start_time
            yield {
                'token': '',
                'done': True,
                'success': True,
                'error': None,
                'full_text': full_text,
                'tokens_generated': tokens_generated,
                'generation_time': elapsed,
                'tokens_per_second': tokens_generated / elapsed if elapsed > 0 else 0,
            }
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                'token': '',
                'done': True,
                'success': False,
                'error': str(e),
                'full_text': full_text if 'full_text' in dir() else '',
            }
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model_loaded and self._llm is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            'provider': self.provider_name,
            'model': self.model_name,
            'available': self.is_available(),
            'model_loaded': self._model_loaded,
            'model_path': self._actual_model_path,
            'repo_id': self._repo_id,
            'filename': self._filename,
            'n_ctx': self._n_ctx,
            'n_gpu_layers': self._n_gpu_layers,
            'n_threads': self._n_threads,
            'use_mmap': self._use_mmap,
            'flash_attn': self._flash_attn,
            'supports_streaming': True,
            'cost_per_token': 0.0,  # Local = free
        }
    
    def get_context_window(self) -> int:
        """Return context window size."""
        return self._n_ctx
