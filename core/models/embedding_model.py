# core/models/embedding_model.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
from utils.logging_config import get_logger

logger = get_logger(__name__)

class EmbeddingModel:
    """
    Wrapper for embedding model with lazy loading and logging.
    Supports both local and future API-based inference.
    """
    
    def __init__(self, config):
        """
        Initialize embedding model.
        
        Args:
            config: ModelConfig instance
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.embedding_dim = None
        
        logger.info(f"EmbeddingModel initialized with {config.embedding_model_name}")
    
    def load(self):
        """Load model and tokenizer."""
        try:
            logger.info("Loading embedding model...")
            
            self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model_name,
                padding_side='left'
            )
            logger.debug(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
            
            # Load model
            try:
                self.model = AutoModel.from_pretrained(
                    self.config.embedding_model_name,
                    attn_implementation="flash_attention_2" if self.config.use_flash_attention else None,
                    torch_dtype=getattr(torch, self.config.torch_dtype),
                    device_map="auto"
                )
                logger.info("Model loaded with Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention failed, falling back to standard: {e}")
                self.model = AutoModel.from_pretrained(
                    self.config.embedding_model_name,
                    device_map="auto"
                )
            
            self.model.eval()
            
            # Get embedding dimension
            self.embedding_dim = self._get_embedding_dim()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
    
    def _get_embedding_dim(self) -> int:
        """Detect embedding dimension by running test inference."""
        try:
            with torch.no_grad():
                test_input = self.tokenizer(
                    ["test"], 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.embedding_max_length, 
                    return_tensors="pt"
                )
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                
                attention_mask = test_input['attention_mask']
                last_hidden_states = test_output.last_hidden_state
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                
                embedding = last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device), 
                    sequence_lengths
                ]
                
                return embedding.shape[1]
        except Exception as e:
            logger.warning(f"Could not detect embedding dim: {e}")
            return 768  # Default fallback
    
    @torch.no_grad()
    def embed(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            logger.debug(f"Embedding {len(texts)} texts")
            
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.embedding_max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # Extract last token embeddings
            attention_mask = inputs['attention_mask']
            last_hidden_states = outputs.last_hidden_state
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            
            embeddings = last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]
            
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            logger.debug(f"Generated embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}", exc_info=True)
            raise
    
    def unload(self):
        """Free model memory."""
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")