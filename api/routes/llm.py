"""
LLM Provider API Routes

Endpoints for managing LLM providers at runtime:
- List providers and models
- Update configuration
- Check status
- Manage API keys
- View usage statistics

File: api/routes/llm.py
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Import logger
try:
    from utils.logger_utils import get_logger
    logger = get_logger("LLMRoutes")
except ImportError:
    import logging
    logger = logging.getLogger("LLMRoutes")

# Import from config
try:
    from config import LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MODEL
except ImportError:
    LLM_PROVIDER = "local"
    OPENROUTER_API_KEY = ""
    OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"


router = APIRouter(prefix="/llm", tags=["LLM"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ProviderInfo(BaseModel):
    """Provider information"""
    id: str
    name: str
    description: str
    requires_api_key: bool
    cost: str


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    name: str = ""
    context_length: int = 0
    pricing: Dict[str, str] = {}
    is_free: bool = False


class LLMConfigUpdate(BaseModel):
    """Request to update LLM configuration"""
    provider: str = Field(..., description="Provider: 'local', 'openrouter', 'none'")
    model: Optional[str] = Field(None, description="Model ID for cloud providers")
    api_key: Optional[str] = Field(None, description="API key (temporary, not stored)")
    save_key: bool = Field(False, description="Whether to save API key securely")


class LLMStatus(BaseModel):
    """LLM provider status response"""
    provider: str
    model: str
    available: bool
    info: Dict[str, Any]


class KeySaveRequest(BaseModel):
    """Request to save API key"""
    provider: str
    api_key: str


class UsageStats(BaseModel):
    """Usage statistics response"""
    session: Dict[str, Any]
    daily: Dict[str, Any] = {}


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/providers", response_model=List[ProviderInfo])
async def list_providers():
    """
    List available LLM providers.
    
    Returns information about each supported provider including
    whether it requires an API key and cost characteristics.
    """
    return [
        ProviderInfo(
            id="local",
            name="Local LLM",
            description="HuggingFace transformers model (GPU required)",
            requires_api_key=False,
            cost="Free (uses local GPU)"
        ),
        ProviderInfo(
            id="openrouter",
            name="OpenRouter",
            description="Cloud API gateway (200+ models including GPT-4, Claude, Gemini)",
            requires_api_key=True,
            cost="Per-token (varies by model, free models available)"
        ),
        ProviderInfo(
            id="none",
            name="None",
            description="RAG-only mode - document retrieval without LLM generation",
            requires_api_key=False,
            cost="Free"
        )
    ]


@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    provider: str,
    api_key: Optional[str] = None,
    free_only: bool = False
):
    """
    List available models for a provider.
    
    For OpenRouter, fetches the full model list from their API.
    For local, returns the configured local model.
    
    Args:
        provider: Provider ID ('openrouter', 'local', 'none')
        api_key: OpenRouter API key (required for openrouter)
        free_only: Only return free models (openrouter only)
    """
    if provider == "openrouter":
        # Get API key from param, stored, or env
        effective_key = api_key
        
        if not effective_key:
            # Try stored key
            try:
                from core.llm_providers.keystore import get_keystore
                effective_key = get_keystore().load_key("openrouter")
            except:
                pass
        
        if not effective_key:
            effective_key = OPENROUTER_API_KEY
        
        if not effective_key:
            raise HTTPException(
                status_code=400,
                detail="OpenRouter API key required. Provide via api_key parameter or environment."
            )
        
        try:
            from core.llm_providers.openrouter import OpenRouterProvider
            
            if free_only:
                models = OpenRouterProvider.get_free_models(effective_key)
            else:
                models = OpenRouterProvider.list_models(effective_key)
            
            return [
                ModelInfo(
                    id=m.get("id", ""),
                    name=m.get("name", m.get("id", "")),
                    context_length=m.get("context_length", 0),
                    pricing=m.get("pricing", {}),
                    is_free=(
                        m.get("pricing", {}).get("prompt", "0") == "0" and
                        m.get("pricing", {}).get("completion", "0") == "0"
                    )
                )
                for m in models
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    elif provider == "local":
        # Return configured local model
        from config import LLM_MODEL
        return [
            ModelInfo(
                id=LLM_MODEL,
                name="Local Model",
                context_length=32768,
                is_free=True
            )
        ]
    
    elif provider == "none":
        return []
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {provider}"
        )


@router.get("/presets")
async def get_model_presets():
    """
    Get recommended model presets.
    
    Returns curated model selections for common use cases,
    prioritizing free models for development.
    """
    return {
        "presets": {
            "free_default": {
                "id": "nvidia/nemotron-3-nano-30b-a3b:free",
                "name": "Nvidia Nemotron (Free)",
                "description": "Fast, free, good for testing"
            },
            "free_google": {
                "id": "google/gemini-2.0-flash-exp:free",
                "name": "Google Gemini Flash (Free)",
                "description": "1M context, experimental"
            },
            "free_openai": {
                "id": "openai/gpt-oss-120b:free",
                "name": "OpenAI GPT OSS (Free)",
                "description": "OpenAI's open model"
            },
            "premium_claude": {
                "id": "anthropic/claude-sonnet-4",
                "name": "Claude Sonnet 4",
                "description": "Best for legal analysis"
            },
            "reasoning": {
                "id": "deepseek/deepseek-r1",
                "name": "DeepSeek R1",
                "description": "Extended reasoning capability"
            }
        },
        "recommended": "free_default"
    }


@router.post("/config")
async def update_config(request: Request, config: LLMConfigUpdate):
    """
    Update LLM provider configuration at runtime.
    
    Switches the active provider and optionally saves the API key.
    Note: Switching to 'local' requires the model to already be loaded.
    """
    logger.info(f"Updating LLM config", {"provider": config.provider})
    
    try:
        from core.llm_providers.factory import LLMProviderFactory
        
        # Prepare kwargs
        kwargs = {}
        if config.api_key:
            kwargs['api_key'] = config.api_key
        if config.model:
            kwargs['model'] = config.model
        
        # If local, don't auto-load (might not have GPU)
        if config.provider == "local":
            kwargs['auto_load'] = False
        
        # Get/create provider
        provider = LLMProviderFactory.get_provider(
            provider_type=config.provider,
            force_reinit=True,
            **kwargs
        )
        
        # Store in app state
        request.app.state.llm_provider = provider
        
        # Save key if requested
        if config.save_key and config.api_key:
            try:
                from core.llm_providers.keystore import get_keystore
                get_keystore().save_key(config.provider, config.api_key)
                logger.info("API key saved securely")
            except Exception as e:
                logger.warning(f"Failed to save key: {e}")
        
        return {
            "success": True,
            "provider": config.provider,
            "model": config.model or provider.model_name,
            "available": provider.is_available()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=LLMStatus)
async def get_status(request: Request):
    """
    Get current LLM provider status.
    
    Returns information about the active provider, model,
    and whether it's ready to serve requests.
    """
    try:
        from core.llm_providers.factory import LLMProviderFactory
        
        provider = LLMProviderFactory.get_current_provider()
        
        if provider is None:
            return LLMStatus(
                provider="none",
                model="none",
                available=False,
                info={"message": "No provider initialized"}
            )
        
        return LLMStatus(
            provider=provider.provider_name,
            model=provider.model_name,
            available=provider.is_available(),
            info=provider.get_info()
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return LLMStatus(
            provider="unknown",
            model="unknown",
            available=False,
            info={"error": str(e)}
        )


@router.post("/keys")
async def save_api_key(request: KeySaveRequest):
    """
    Save an API key securely.
    
    Uses encrypted storage with machine-specific key derivation.
    """
    try:
        from core.llm_providers.keystore import get_keystore
        
        keystore = get_keystore()
        
        # Validate key if OpenRouter
        if request.provider == "openrouter":
            from core.llm_providers.openrouter import OpenRouterProvider
            if not OpenRouterProvider.validate_api_key(request.api_key):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid OpenRouter API key"
                )
        
        success = keystore.save_key(request.provider, request.api_key)
        
        if success:
            return {"success": True, "message": f"API key saved for {request.provider}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save key")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/keys/{provider}")
async def delete_api_key(provider: str):
    """Delete a stored API key"""
    try:
        from core.llm_providers.keystore import get_keystore
        
        keystore = get_keystore()
        deleted = keystore.delete_key(provider)
        
        return {
            "success": deleted,
            "message": f"Key {'deleted' if deleted else 'not found'} for {provider}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keys")
async def list_stored_keys():
    """List providers with stored API keys (not the keys themselves)"""
    try:
        from core.llm_providers.keystore import get_keystore
        
        keystore = get_keystore()
        providers = keystore.list_providers()
        
        return {
            "providers_with_keys": providers
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/usage", response_model=UsageStats)
async def get_usage_stats():
    """Get token usage statistics"""
    try:
        from core.llm_providers.usage_tracker import get_usage_tracker
        
        tracker = get_usage_tracker()
        
        return UsageStats(
            session=tracker.get_session_stats(),
            daily=tracker.get_daily_stats(days=7)
        )
        
    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        return UsageStats(session={}, daily={})


@router.post("/cache/clear")
async def clear_cache():
    """Clear the response cache"""
    try:
        from core.llm_providers.cache import get_response_cache
        
        cache = get_response_cache()
        stats_before = cache.get_stats()
        cache.clear()
        
        return {
            "success": True,
            "entries_cleared": stats_before.get("size", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """Get response cache statistics"""
    try:
        from core.llm_providers.cache import get_response_cache
        
        cache = get_response_cache()
        return cache.get_stats()
        
    except Exception as e:
        return {"error": str(e)}
