"""
Local LLM Provider - HuggingFace models with quantization support
"""

from typing import Dict, Any, Optional, Generator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import BaseLLMProvider
from logger_utils import get_logger

logger = get_logger(__name__)


class LocalLLMProvider(BaseLLMProvider):
    """
    Local inference provider using HuggingFace Transformers

    Supports:
    - CPU/GPU device placement
    - 4-bit and 8-bit quantization
    - Full precision inference
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Import config defaults
        from config import (
            LLM_MODEL, LLM_DEVICE, LLM_LOAD_IN_4BIT, LLM_LOAD_IN_8BIT
        )

        self.model_name = self.config.get('model', LLM_MODEL)
        self.device = self.config.get('device', LLM_DEVICE)
        self.load_in_4bit = self.config.get('load_in_4bit', LLM_LOAD_IN_4BIT)
        self.load_in_8bit = self.config.get('load_in_8bit', LLM_LOAD_IN_8BIT)

        self.model = None
        self.tokenizer = None

    def initialize(self) -> bool:
        """Load model with specified configuration"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading local model: {self.model_name}")
            logger.info(f"Device: {self.device}, 4bit: {self.load_in_4bit}, 8bit: {self.load_in_8bit}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Configure quantization
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                logger.info("Using 8-bit quantization")

            # Load model
            load_kwargs = {
                'trust_remote_code': True,
                'device_map': 'auto' if self.device == 'cuda' else None,
            }

            if quantization_config:
                load_kwargs['quantization_config'] = quantization_config
            elif self.device == 'cpu':
                load_kwargs['torch_dtype'] = torch.float32
            else:
                load_kwargs['torch_dtype'] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            if self.device == 'cpu' and not quantization_config:
                self.model = self.model.to('cpu')

            self._initialized = True
            logger.info(f"Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using local model"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get('top_p', 0.9),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generation using TextIteratorStreamer"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = {
                **inputs,
                'streamer': streamer,
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': kwargs.get('top_p', 0.9),
                'do_sample': temperature > 0,
                'pad_token_id': self.tokenizer.eos_token_id
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for text in streamer:
                yield text

            thread.join()

        except ImportError:
            # Fallback to non-streaming
            yield self.generate(prompt, max_tokens, temperature, **kwargs)

    def shutdown(self) -> None:
        """Free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        super().shutdown()
        logger.info("Local provider shutdown complete")
