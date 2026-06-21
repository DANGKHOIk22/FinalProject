import logging
from pathlib import Path

import yaml
import litellm
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router

# Enable dropping unsupported parameters (e.g., temperature/top_p for reasoning models)
litellm.drop_params = True

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"

# Shared across all callers — models are stateless so this is safe.
_router: Router | None = None


def _get_router() -> Router:
    global _router
    if _router is None:
        config = yaml.safe_load(_CONFIG_PATH.read_text())
        router_settings = config.get("router_settings", {})
        _router = Router(
            model_list=config["model_list"],
            fallbacks=router_settings.get("fallbacks", []),
        )
    return _router


def get_llm(role: str = "tool") -> ChatLiteLLMRouter:
    """Return a ChatLiteLLMRouter for the given role."""
    if role in ["aggregator"]:
        # Enable thinking for some roles to enhance UX
        return ChatLiteLLMRouter(router=_get_router(), model_name=role,num_retries=0, streaming=True)
    return ChatLiteLLMRouter(router=_get_router(), model_name=role,num_retries=0, streaming=False)


async def warm_llm(llm, system_message, label: str) -> None:
    """Push a system prompt at max_completion_tokens=1 to warm the HTTP pool and prime
    the prompt cache. Non-fatal — logs a warning on failure."""
    try:
        await llm.ainvoke(
            [system_message, HumanMessage(content="warmup")],
            max_completion_tokens=1,
        )
        logger.info(f"✅ {label} warmed up")
    except Exception as e:
        logger.warning(f"⚠️ {label} warmup failed (non-fatal): {e}")
