"""
Encrypted API Key Storage

Securely stores API keys using Fernet encryption with
machine-specific key derivation.

File: core/llm_providers/keystore.py
"""

import os
import json
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict

# Try to import cryptography, fall back to simple obfuscation if not available
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# Import logger
try:
    from utils.logger_utils import get_logger
    logger = get_logger("SecureKeyStore")
except ImportError:
    import logging
    logger = logging.getLogger("SecureKeyStore")


class SecureKeyStore:
    """
    Encrypted storage for API keys.
    
    Uses Fernet symmetric encryption with a key derived from:
    - Machine hostname
    - Username
    - A salt stored alongside the encrypted data
    
    Falls back to base64 obfuscation if cryptography is not available.
    
    Storage location: ~/.legal_rag/keys.json (encrypted)
    """
    
    DEFAULT_DIR = Path.home() / ".legal_rag"
    KEYS_FILE = "keys.json"
    
    def __init__(self, storage_dir: Path = None):
        """
        Initialize key store.
        
        Args:
            storage_dir: Directory for key storage (default: ~/.legal_rag)
        """
        self.storage_dir = storage_dir or self.DEFAULT_DIR
        self.keys_path = self.storage_dir / self.KEYS_FILE
        
        # Create storage directory if needed
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate encryption key
        self._fernet = self._create_fernet()
        
        if CRYPTOGRAPHY_AVAILABLE:
            logger.info("Secure key store initialized with encryption")
        else:
            logger.warning("cryptography not available, using obfuscation only")
    
    def _get_machine_identifier(self) -> bytes:
        """Generate machine-specific identifier for key derivation"""
        import socket
        import getpass
        
        hostname = socket.gethostname()
        username = getpass.getuser()
        
        # Combine machine-specific info
        identifier = f"{hostname}:{username}:legal_rag_keystore"
        return identifier.encode('utf-8')
    
    def _create_fernet(self) -> Optional['Fernet']:
        """Create Fernet cipher with machine-specific key"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return None
        
        # Use machine identifier as password
        password = self._get_machine_identifier()
        
        # Use fixed salt for consistency (stored in code)
        # This is acceptable because the password itself is machine-specific
        salt = b'legal_rag_secure_keystore_v1'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string"""
        if self._fernet:
            encrypted = self._fernet.encrypt(plaintext.encode('utf-8'))
            return encrypted.decode('utf-8')
        else:
            # Fallback: base64 obfuscation (not secure, just obscured)
            return base64.b64encode(plaintext.encode('utf-8')).decode('utf-8')
    
    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a string"""
        if self._fernet:
            try:
                decrypted = self._fernet.decrypt(ciphertext.encode('utf-8'))
                return decrypted.decode('utf-8')
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                return ""
        else:
            # Fallback: base64 decode
            try:
                return base64.b64decode(ciphertext.encode('utf-8')).decode('utf-8')
            except:
                return ""
    
    def _load_keys(self) -> Dict[str, str]:
        """Load encrypted keys from file"""
        if not self.keys_path.exists():
            return {}
        
        try:
            with open(self.keys_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            return {}
    
    def _save_keys(self, keys: Dict[str, str]):
        """Save encrypted keys to file"""
        try:
            with open(self.keys_path, 'w') as f:
                json.dump(keys, f, indent=2)
            
            # Restrict file permissions on Unix
            if os.name != 'nt':  # Not Windows
                os.chmod(self.keys_path, 0o600)
                
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
    
    def save_key(self, provider: str, api_key: str) -> bool:
        """
        Encrypt and save API key.
        
        Args:
            provider: Provider name (e.g., 'openrouter')
            api_key: API key to store
            
        Returns:
            True if successful
        """
        if not api_key:
            logger.warning("Cannot save empty API key")
            return False
        
        try:
            keys = self._load_keys()
            keys[provider] = self._encrypt(api_key)
            self._save_keys(keys)
            
            logger.info(f"Saved encrypted API key for {provider}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save key: {e}")
            return False
    
    def load_key(self, provider: str) -> Optional[str]:
        """
        Load and decrypt API key.
        
        Args:
            provider: Provider name
            
        Returns:
            Decrypted API key or None
        """
        keys = self._load_keys()
        
        if provider not in keys:
            return None
        
        decrypted = self._decrypt(keys[provider])
        
        if decrypted:
            logger.debug(f"Loaded API key for {provider}")
        
        return decrypted if decrypted else None
    
    def delete_key(self, provider: str) -> bool:
        """
        Delete stored API key.
        
        Args:
            provider: Provider name
            
        Returns:
            True if deleted
        """
        keys = self._load_keys()
        
        if provider in keys:
            del keys[provider]
            self._save_keys(keys)
            logger.info(f"Deleted API key for {provider}")
            return True
        
        return False
    
    def has_key(self, provider: str) -> bool:
        """Check if key exists for provider"""
        keys = self._load_keys()
        return provider in keys
    
    def list_providers(self) -> list:
        """List providers with stored keys"""
        keys = self._load_keys()
        return list(keys.keys())
    
    def clear_all(self):
        """Delete all stored keys"""
        if self.keys_path.exists():
            self.keys_path.unlink()
            logger.info("All API keys cleared")


# Global instance
_keystore: Optional[SecureKeyStore] = None


def get_keystore() -> SecureKeyStore:
    """Get global keystore instance"""
    global _keystore
    if _keystore is None:
        _keystore = SecureKeyStore()
    return _keystore
