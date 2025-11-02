# examples/llm_examples.py
"""
Examples of using the LLM wrapper system.
"""
import os
from pathlib import Path
from core.models.llm_factory import LLMFactory
from core.models.base_llm import GenerationConfig
from config.llm_config import get_preset
from utils.logging_config import get_logger

logger = get_logger('examples')

def example_1_local_model():
    """Example 1: Using local HuggingFace model."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Local HuggingFace Model")
    logger.info("=" * 80)
    
    # Create local LLM
    llm = LLMFactory.create_llm(
        llm_type='local',
        model_name='Azzindani/Deepseek_ID_Legal_Preview',
        device='cuda',
        load_in_4bit=True,  # Use 4-bit for efficiency
        auto_load=True
    )
    
    logger.info(f"✅ Model loaded: {llm}")
    
    # Generate text
    prompt = "Apa yang dimaksud dengan hukum pidana?"
    
    logger.info(f"Prompt: {prompt}")
    
    config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9
    )
    
    response = llm.generate(prompt, config)
    
    logger.info(f"Response: {response.text[:200]}...")
    logger.info(f"Tokens used: {response.tokens_used}")
    
    # Get statistics
    stats = llm.get_stats()
    logger.info(f"Stats: {stats}")
    
    # Cleanup
    llm.unload()


def example_2_streaming_generation():
    """Example 2: Streaming generation."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Streaming Generation")
    logger.info("=" * 80)
    
    llm = LLMFactory.create_llm(
        'local',
        'Azzindani/Deepseek_ID_Legal_Preview',
        device='cuda',
        auto_load=True
    )
    
    prompt = "Jelaskan prosedur pembuatan peraturan daerah."
    
    logger.info(f"Prompt: {prompt}")
    logger.info("Streaming response:")
    
    config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        stream=True
    )
    
    full_response = ""
    for chunk in llm.generate_stream(prompt, config):
        print(chunk, end='', flush=True)
        full_response += chunk
    
    print("\n")
    logger.info(f"Complete response length: {len(full_response)} chars")
    
    llm.unload()


def example_3_openai_api():
    """Example 3: Using OpenAI API."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: OpenAI API")
    logger.info("=" * 80)
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        logger.warning("⚠️ OPENAI_API_KEY not set, skipping example")
        return
    
    # Create OpenAI LLM
    llm = LLMFactory.create_llm(
        'openai',
        'gpt-3.5-turbo',
        api_key=api_key
    )
    
    logger.info(f"✅ OpenAI LLM created: {llm}")
    
    prompt = "Explain the concept of rule of law in 2 sentences."
    
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.7
    )
    
    response = llm.generate(prompt, config)
    
    logger.info(f"Response: {response.text}")
    logger.info(f"Tokens: {response.tokens_used}")
    logger.info(f"Model: {response.model}")


def example_4_using_presets():
    """Example 4: Using configuration presets."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Using Configuration Presets")
    logger.info("=" * 80)
    
    # Get preset configuration
    preset = get_preset('deepseek_id_legal')
    
    logger.info(f"Using preset: {preset.model_name}")
    
    # Create LLM from preset
    llm = LLMFactory.create_llm(**preset.to_dict(), auto_load=False)
    
    logger.info(f"✅ LLM created from preset: {llm}")
    
    # Check availability without loading
    logger.info(f"Available: {llm.is_available()}")


def example_5_error_handling():
    """Example 5: Error handling and retries."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Error Handling")
    logger.info("=" * 80)
    
    # Simulate API with occasional failures
    api_key = os.getenv('OPENAI_API_KEY', 'dummy-key')
    
    llm = LLMFactory.create_llm(
        'openai',
        'gpt-3.5-turbo',
        api_key=api_key
    )
    
    prompt = "Test prompt"
    config = GenerationConfig(max_new_tokens=50)
    
    try:
        response = llm.generate(prompt, config)
        logger.info(f"✅ Success: {response.finish_reason}")
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
    
    # Check stats
    stats = llm.get_stats()
    logger.info(f"Success rate: {stats['success_rate']:.1f}%")


def example_6_comparing_models():
    """Example 6: Comparing local vs API models."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 6: Model Comparison")
    logger.info("=" * 80)
    
    prompt = "What is justice?"
    config = GenerationConfig(max_new_tokens=100, temperature=0.7)
    
    models = []
    
    # Local model
    try:
        local_llm = LLMFactory.create_llm(
            'local',
            'gpt2',  # Small model for testing
            device='cpu',
            auto_load=True
        )
        models.append(('Local GPT-2', local_llm))
    except Exception as e:
        logger.warning(f"Could not load local model: {e}")
    
    # API model (if available)
    if os.getenv('OPENAI_API_KEY'):
        try:
            api_llm = LLMFactory.create_llm(
                'openai',
                'gpt-3.5-turbo',
                api_key=os.getenv('OPENAI_API_KEY')
            )
            models.append(('OpenAI GPT-3.5', api_llm))
        except Exception as e:
            logger.warning(f"Could not create API model: {e}")
    
    # Compare
    for model_name, llm in models:
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'='*40}")
        
        try:
            import time
            start = time.time()
            response = llm.generate(prompt, config)
            elapsed = time.time() - start
            
            logger.info(f"Response: {response.text[:100]}...")
            logger.info(f"Time: {elapsed:.2f}s")
            logger.info(f"Tokens: {response.tokens_used}")
            
        except Exception as e:
            logger.error(f"Failed: {e}")


if __name__ == '__main__':
    import sys
    
    examples = {
        '1': example_1_local_model,
        '2': example_2_streaming_generation,
        '3': example_3_openai_api,
        '4': example_4_using_presets,
        '5': example_5_error_handling,
        '6': example_6_comparing_models
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Available examples:")
        print("  python examples/llm_examples.py 1  # Local model")
        print("  python examples/llm_examples.py 2  # Streaming")
        print("  python examples/llm_examples.py 3  # OpenAI API")
        print("  python examples/llm_examples.py 4  # Using presets")
        print("  python examples/llm_examples.py 5  # Error handling")
        print("  python examples/llm_examples.py 6  # Model comparison")