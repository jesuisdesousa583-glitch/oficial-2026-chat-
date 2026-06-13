"""
Script de teste para Ollama + OpenAI fallback.
Valida a integração antes do deploy no Render.

Uso:
    $ python backend/test_ollama_integration.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from ollama_config import (
    ollama_health_check,
    generate_image_ollama,
    generate_image_openai,
    generate_image,
    get_image_generation_status,
    OLLAMA_ENABLED,
    OLLAMA_URL,
    EMERGENT_LLM_KEY,
)


async def test_ollama_health():
    """Testa conectividade com Ollama."""
    print("\n" + "="*60)
    print("🔍 TEST 1: Ollama Health Check")
    print("="*60)
    
    status = await ollama_health_check()
    print(f"✓ Ollama Health: {status['ok']}")
    if status['ok']:
        print(f"  - URL: {status['url']}")
        print(f"  - Model: {status['model']}")
        print(f"  - Available models: {status.get('available_models', [])[:3]}")
    else:
        print(f"  ⚠️  Error: {status['error']}")
    
    return status['ok']


async def test_image_generation_unified():
    """Testa geração de imagem com fallback automático."""
    print("\n" + "="*60)
    print("🎨 TEST 2: Unified Image Generation (Ollama + OpenAI Fallback)")
    print("="*60)
    
    prompt = "Professional modern social media graphic for a Brazilian law firm. Style: clean minimal sophisticated. No people, just abstract geometric shapes in navy and amber."
    
    print(f"Prompt: {prompt[:80]}...")
    print("Generating image with fallback strategy...")
    
    img = await generate_image(prompt, size="512x512", quality="standard")
    
    if img:
        print(f"✓ Image generated successfully!")
        print(f"  - Size: {len(img)} bytes")
        print(f"  - Format: PNG")
        
        # Save for inspection
        output_path = Path(__file__).parent / "test_output_image.png"
        with open(output_path, "wb") as f:
            f.write(img)
        print(f"  - Saved to: {output_path}")
        return True
    else:
        print("✗ Image generation failed (no provider available)")
        return False


async def test_status_endpoint():
    """Testa endpoint de status."""
    print("\n" + "="*60)
    print("📊 TEST 3: Status Endpoint")
    print("="*60)
    
    status = await get_image_generation_status()
    
    print(f"Timestamp: {status['timestamp']}")
    print(f"\nOllama:")
    print(f"  - Enabled: {status['ollama']['enabled']}")
    print(f"  - URL: {status['ollama'].get('url', 'N/A')}")
    print(f"  - Connected: {status['ollama'].get('ok', False)}")
    
    print(f"\nOpenAI:")
    print(f"  - Enabled: {status['openai']['enabled']}")
    print(f"  - Has Key: {status['openai']['has_key']}")
    if status['openai']['key_masked']:
        print(f"  - Key: {status['openai']['key_masked']}")
    
    return True


async def test_render_deployment_config():
    """Valida configuração para deploy no Render."""
    print("\n" + "="*60)
    print("🚀 TEST 4: Render Deployment Configuration")
    print("="*60)
    
    checks = {
        "OLLAMA_ENABLED": OLLAMA_ENABLED,
        "OLLAMA_URL set": bool(OLLAMA_URL),
        "EMERGENT_LLM_KEY set": bool(EMERGENT_LLM_KEY),
    }
    
    all_ok = True
    for check, result in checks.items():
        status_icon = "✓" if result else "✗"
        print(f"{status_icon} {check}: {result}")
        if not result:
            all_ok = False
    
    if all_ok:
        print("\n✓ Configuration ready for Render deployment!")
    else:
        print("\n⚠️  Missing configuration. See instructions below.")
        print_deployment_instructions()
    
    return all_ok


def print_deployment_instructions():
    """Imprime instruções de deployment."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║         CONFIGURAÇÃO PARA DEPLOY NO RENDER                    ║
╚════════════════════════════════════════════════════════════════╝

1. Criar Web Service Ollama (opcional):
   - Dashboard Render → New + → Web Service
   - Runtime: Docker
   - Dockerfile simples que use ollama/ollama:latest
   - Nome: kenia-ollama
   - Environment:
     OLLAMA_KEEP_ALIVE=1h
     OLLAMA_NUM_THREADS=4

2. Backend Service Variables:
   ✓ OLLAMA_ENABLED=true (ou false para usar só OpenAI)
   ✓ OLLAMA_URL=https://kenia-ollama.onrender.com
   ✓ OLLAMA_TIMEOUT=120
   ✓ EMERGENT_LLM_KEY=sk-... (sua chave OpenAI/Emergent)
   ✓ USE_OPENAI_FALLBACK=true

3. Testar health check:
   GET https://kenia-backend.onrender.com/api/creatives/image-status

4. Testar geração:
   POST https://kenia-backend.onrender.com/api/creatives/generate
   {
     "title": "Test",
     "network": "instagram",
     "format": "post",
     "topic": "Law firm branding",
     "tone": "professional",
     "case_type": "Geral"
   }
""")


async def run_all_tests():
    """Executa todos os testes."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     OLLAMA + OPENAI IMAGE GENERATION INTEGRATION TESTS        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    # Test 1: Ollama health
    results["Health Check"] = await test_ollama_health()
    
    # Test 2: Image generation
    results["Image Generation"] = await test_image_generation_unified()
    
    # Test 3: Status endpoint
    results["Status Endpoint"] = await test_status_endpoint()
    
    # Test 4: Deployment config
    results["Deployment Config"] = await test_render_deployment_config()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        icon = "✓" if result else "✗"
        print(f"{icon} {test}")
    
    print(f"\nResult: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ All tests passed! Ready for Render deployment.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review configuration above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
