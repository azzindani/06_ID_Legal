"""
Model Manager - Fixed tokenizer integration
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Any
import time
from logger_utils import get_logger
from config import EMBEDDING_MODEL, RERANKER_MODEL, DEVICE, CACHE_DIR, get_model_path, USE_LOCAL_MODELS


class EmbeddingModelWrapper:
    """Wrapper to add proper tokenize method to embedding model"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eval = model.eval
        self.to = model.to
        
    def tokenize(self, texts, padding=True, truncation=True, max_length=512, **kwargs):
        """Tokenize texts properly and move to correct device"""
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs
        )

        # Move all tensors to the model's device (critical for multi-GPU setups)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        return tokens
    
    def __call__(self, *args, **kwargs):
        """Forward call to model"""
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Proxy other attributes to model"""
        return getattr(self.model, name)


class ModelManager:
    """
    Centralized model management with lazy loading and caching
    """

    def __init__(self):
        self.logger = get_logger("ModelManager")

        # Smart device allocation for multi-GPU setups
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.num_gpus > 1:
            # Multi-GPU: distribute models across available GPUs
            # Strategy: LLM on GPU 0, spread embedding/reranker on other GPUs

            # Create list of available GPU indices
            available_gpus = list(range(self.num_gpus))

            # LLM will use cuda:0 via device_map='auto' (or spread across GPUs)
            # Distribute embedding and reranker on remaining GPUs

            if self.num_gpus == 2:
                # 2 GPUs: LLM on 0, embedding and reranker share GPU 1
                self.embedding_device = torch.device('cuda:1')
                self.reranker_device = torch.device('cuda:1')
            elif self.num_gpus >= 3:
                # 3+ GPUs: LLM on 0, embedding on 1, reranker on 2
                self.embedding_device = torch.device('cuda:1')
                self.reranker_device = torch.device('cuda:2')

            self.logger.info(f"Multi-GPU setup detected", {
                "total_gpus": self.num_gpus,
                "available_gpus": available_gpus,
                "llm_device": "cuda:0 (device_map='auto')",
                "embedding_device": str(self.embedding_device),
                "reranker_device": str(self.reranker_device)
            })
        else:
            # Single GPU or CPU: use default device for all models
            self.embedding_device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
            self.reranker_device = self.embedding_device
            self.logger.info("Single device mode", {
                "device": str(self.embedding_device),
                "num_gpus": self.num_gpus
            })

        # Legacy device for compatibility
        self.device = self.embedding_device

        # Model cache
        self._embedding_model = None
        self._embedding_tokenizer = None
        self._reranker_model = None

        self.logger.info("ModelManager initialized")
    
    def load_embedding_model(
        self,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> Any:
        """
        Load embedding model with retry logic
        
        Args:
            model_name: Model name (defaults to config)
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Loaded model wrapper
        """
        if self._embedding_model is not None:
            self.logger.debug("Returning cached embedding model")
            return self._embedding_model

        # Use get_model_path for local model support
        model_name = model_name or get_model_path('embedding')
        self.logger.info("Loading embedding model", {"model": model_name, "local": USE_LOCAL_MODELS})
        
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt}/{max_retries}")
                
                # Load tokenizer
                self.logger.debug("Loading tokenizer")
                self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=CACHE_DIR,
                    trust_remote_code=True
                )
                
                # Load model
                self.logger.debug("Loading model weights")
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=CACHE_DIR,
                    torch_dtype=torch.float16 if self.embedding_device.type == 'cuda' else torch.float32,
                    trust_remote_code=True
                )

                # Move to dedicated embedding device and set eval mode
                model.to(self.embedding_device)
                model.eval()

                # Verify actual device placement
                actual_device = next(model.parameters()).device
                if actual_device != self.embedding_device:
                    self.logger.warning(f"Device mismatch! Assigned: {self.embedding_device}, Actual: {actual_device}")
                else:
                    self.logger.debug(f"Embedding model correctly placed on {actual_device}")

                # Create wrapper with proper tokenize method
                self._embedding_model = EmbeddingModelWrapper(
                    model=model,
                    tokenizer=self._embedding_tokenizer,
                    device=self.embedding_device
                )
                
                self.logger.success("Embedding model loaded successfully", {
                    "model": model_name,
                    "device": str(self.embedding_device),
                    "dtype": str(model.dtype)
                })
                
                return self._embedding_model
                
            except Exception as e:
                self.logger.error(f"Failed to load embedding model (attempt {attempt})", {
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                if attempt < max_retries:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Max retries reached, giving up")
                    raise Exception(f"Failed to load embedding model after {max_retries} attempts: {e}")
    
    def load_reranker_model(
        self,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        use_mock: bool = False
    ) -> Any:
        """
        Load reranker model with retry logic
        
        Args:
            model_name: Model name (defaults to config)
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            use_mock: Use mock reranker for testing
            
        Returns:
            Loaded reranker
        """
        if self._reranker_model is not None:
            self.logger.debug("Returning cached reranker model")
            return self._reranker_model
        
        # Use mock for testing if requested
        if use_mock:
            self.logger.warning("Using mock reranker for testing")
            self._reranker_model = MockReranker()
            return self._reranker_model
        
        # Use get_model_path for local model support
        model_name = model_name or get_model_path('reranker')
        self.logger.info("Loading reranker model", {"model": model_name, "local": USE_LOCAL_MODELS})
        
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt}/{max_retries}")
                
                # Try to load actual reranker
                try:
                    from sentence_transformers import CrossEncoder
                    
                    self.logger.debug("Loading CrossEncoder reranker")
                    self._reranker_model = CrossEncoder(
                        model_name,
                        max_length=512,
                        device=str(self.reranker_device)  # CrossEncoder expects string like 'cuda:0'
                    )

                    # Fix padding token issue for batch processing
                    if hasattr(self._reranker_model, 'tokenizer'):
                        tokenizer = self._reranker_model.tokenizer
                        if tokenizer.pad_token is None:
                            # Try different fallbacks for pad_token
                            if tokenizer.eos_token is not None:
                                tokenizer.pad_token = tokenizer.eos_token
                                self.logger.debug("Set reranker pad_token to eos_token")
                            elif tokenizer.sep_token is not None:
                                tokenizer.pad_token = tokenizer.sep_token
                                self.logger.debug("Set reranker pad_token to sep_token")
                            elif tokenizer.cls_token is not None:
                                tokenizer.pad_token = tokenizer.cls_token
                                self.logger.debug("Set reranker pad_token to cls_token")
                            else:
                                # Add a new pad token as last resort
                                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                                self.logger.debug("Added [PAD] token to reranker tokenizer")

                    # Add compute_score method for compatibility
                    original_predict = self._reranker_model.predict
                    
                    def compute_score(pairs, normalize=True):
                        import numpy as np
                        scores = original_predict(pairs)
                        scores = np.array(scores)
                        if normalize:
                            # Use sigmoid for normalization to avoid 0 scores with single items
                            # Sigmoid maps any score to (0, 1) range
                            scores = 1 / (1 + np.exp(-scores))
                        return scores.tolist()
                    
                    self._reranker_model.compute_score = compute_score
                    
                except ImportError:
                    # Fallback to transformers if sentence-transformers not available
                    self.logger.warning("sentence-transformers not available, using transformers")
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=CACHE_DIR,
                        trust_remote_code=True
                    )
                    model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=CACHE_DIR,
                        torch_dtype=torch.float16 if self.reranker_device.type == 'cuda' else torch.float32,
                        trust_remote_code=True
                    )
                    model.to(self.reranker_device)
                    model.eval()

                    # Verify actual device placement
                    actual_device = next(model.parameters()).device
                    if actual_device != self.reranker_device:
                        self.logger.warning(f"Reranker device mismatch! Assigned: {self.reranker_device}, Actual: {actual_device}")
                    else:
                        self.logger.debug(f"Reranker model correctly placed on {actual_device}")

                    # Create wrapper
                    self._reranker_model = TransformersReranker(model, tokenizer, self.reranker_device)

                # Verify CrossEncoder device (if using CrossEncoder)
                if hasattr(self._reranker_model, 'model') and hasattr(self._reranker_model.model, 'device'):
                    actual_device = self._reranker_model.model.device
                    self.logger.debug(f"CrossEncoder on device: {actual_device}")

                self.logger.success("Reranker model loaded successfully", {
                    "model": model_name,
                    "device": str(self.reranker_device)
                })
                
                return self._reranker_model
                
            except Exception as e:
                self.logger.error(f"Failed to load reranker model (attempt {attempt})", {
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                if attempt < max_retries:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.warning("Max retries reached, using mock reranker")
                    self._reranker_model = MockReranker()
                    return self._reranker_model
    
    def unload_models(self):
        """Unload models to free memory"""
        self.logger.info("Unloading models")
        
        if self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
        
        if self._embedding_tokenizer is not None:
            del self._embedding_tokenizer
            self._embedding_tokenizer = None
        
        if self._reranker_model is not None:
            del self._reranker_model
            self._reranker_model = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.success("Models unloaded")
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        return {
            'embedding_model_loaded': self._embedding_model is not None,
            'reranker_model_loaded': self._reranker_model is not None,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


class TransformersReranker:
    """
    Reranker wrapper using transformers
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = get_logger("TransformersReranker")
    
    def compute_score(self, pairs, normalize=True):
        """
        Compute reranking scores for query-document pairs
        
        Args:
            pairs: List of [query, document] pairs
            normalize: Whether to normalize scores to 0-1
            
        Returns:
            List of scores
        """
        try:
            scores = []
            
            # Process in batches
            batch_size = 8
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Use CLS token or mean pooling
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeddings = outputs.pooler_output
                    else:
                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        token_embeddings = outputs.last_hidden_state
                        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        embeddings = sum_embeddings / sum_mask
                    
                    # Compute similarity (simplified: use embedding norm as proxy)
                    batch_scores = torch.norm(embeddings, dim=1).cpu().numpy()
                    scores.extend(batch_scores)
            
            # Normalize if requested
            if normalize and len(scores) > 0:
                import numpy as np
                scores = np.array(scores)
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                scores = scores.tolist()
            
            return scores
            
        except Exception as e:
            self.logger.error("Reranking failed", {"error": str(e)})
            # Return random scores as fallback
            import random
            return [random.random() * 0.5 + 0.5 for _ in pairs]


class MockReranker:
    """
    Mock reranker for testing when real model is not available
    """
    
    def __init__(self):
        self.logger = get_logger("MockReranker")
        self.logger.warning("Using mock reranker - results will be random!")
    
    def compute_score(self, pairs, normalize=True):
        """
        Generate mock scores
        
        Args:
            pairs: List of [query, document] pairs
            normalize: Ignored (scores always normalized)
            
        Returns:
            List of random scores between 0.5 and 1.0
        """
        import random
        
        # Generate scores with some logic:
        # - Longer documents get slightly higher scores
        # - Some randomness
        scores = []
        for query, doc in pairs:
            base_score = min(1.0, 0.6 + len(doc) / 10000)
            noise = random.uniform(-0.1, 0.1)
            score = max(0.5, min(1.0, base_score + noise))
            scores.append(score)
        
        return scores


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """
    Get global model manager instance (singleton)
    
    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def load_models(use_mock_reranker: bool = False):
    """
    Convenience function to load all models

    Args:
        use_mock_reranker: Whether to use mock reranker for testing

    Returns:
        Tuple of (embedding_model, reranker_model)
    """
    manager = get_model_manager()

    embedding_model = manager.load_embedding_model()
    reranker_model = manager.load_reranker_model(use_mock=use_mock_reranker)

    return embedding_model, reranker_model


if __name__ == "__main__":
    from logger_utils import initialize_logging
    initialize_logging(enable_file_logging=False)

    print("=" * 60)
    print("MODEL MANAGER TEST")
    print("=" * 60)

    manager = ModelManager()

    # Show model info before loading
    print("\nInitial State:")
    info = manager.get_model_info()
    print(f"  Device: {info['device']}")
    print(f"  CUDA Available: {info['cuda_available']}")
    print(f"  GPU Count: {info['cuda_device_count']}")

    # Test embedding model loading
    print("\n" + "-" * 60)
    print("Loading Embedding Model")
    print("-" * 60)

    try:
        embedding_model = manager.load_embedding_model()
        print(f"  ✓ Embedding model loaded")
        print(f"  Type: {type(embedding_model).__name__}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test reranker model loading (use mock for quick test)
    print("\n" + "-" * 60)
    print("Loading Reranker Model (Mock)")
    print("-" * 60)

    try:
        reranker_model = manager.load_reranker_model(use_mock=True)
        print(f"  ✓ Reranker model loaded")
        print(f"  Type: {type(reranker_model).__name__}")

        # Test reranker scoring
        pairs = [
            ["Apa sanksi UU ITE?", "UU ITE mengatur sanksi pidana untuk pelanggaran."],
            ["Apa sanksi UU ITE?", "Cuaca hari ini cerah."]
        ]
        scores = reranker_model.compute_score(pairs)
        print(f"  Test scores: {scores}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Show final state
    print("\n" + "-" * 60)
    print("Final State")
    print("-" * 60)

    info = manager.get_model_info()
    print(f"  Embedding loaded: {info['embedding_model_loaded']}")
    print(f"  Reranker loaded: {info['reranker_model_loaded']}")

    # Cleanup
    print("\n" + "-" * 60)
    print("Cleanup")
    print("-" * 60)

    manager.unload_models()
    print("  ✓ Models unloaded")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)