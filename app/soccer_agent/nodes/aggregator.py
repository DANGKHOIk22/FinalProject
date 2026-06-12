import logging
from datetime import datetime
from langgraph.graph.state import RunnableConfig
from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.prompts.agent import get_aggregator_prompt_template

logger = logging.getLogger(__name__)

class AggregatorNode:
    def __init__(self, aggregator_llm):
        self.aggregator_llm = aggregator_llm

    async def aggregator_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """Aggregate results from parallel workers and handle pending clarifications."""
        from langchain_core.messages import AIMessage

        messages = state.get("messages", [])
        user_query = messages[-1].text if messages else ""
        image_id_list = (state.get("additional_material") or {}).get("image_id") or []
        additional_material = ", ".join(image_id_list) if image_id_list else "None"
        conversation_history = state.get("conversation_history", "No previous conversation.")
        results = state.get("parallel_results", [])
        pending_clarifications = state.get("pending_clarifications") or []

        logger.info("="*70)
        logger.info("🧠 Starting AGGREGATOR STEP")
        logger.info(f"   worker_results={len(results)}, pending_clarifications={len(pending_clarifications)}")

        # Build clarification block (appended to every response when present)
        clarification_block = ""
        if pending_clarifications:
            questions = "\n".join(f"- {q}" for q in pending_clarifications)
            clarification_block = f"\n\n---\nNgoài ra, tôi cần thêm thông tin để trả lời đầy đủ:\n{questions}"

        # No worker results at all → only clarifications
        if not results:
            logger.info("⚡ No worker results — returning clarification questions only.")
            return {
                "messages": [AIMessage(content=(
                    "Câu hỏi của bạn chưa đủ rõ ràng để tôi trả lời chính xác. "
                    "Bạn có thể làm rõ thêm không?" + clarification_block
                ))],
            }

        # Single worker result → bypass LLM, append clarifications if any
        if len(results) == 1 and not pending_clarifications:
            logger.info("⚡ Single worker result — bypassing aggregator LLM.")
            return {"messages": [AIMessage(content=results[0])]}

        if results:
            worker_results_str = "\n".join([f"Worker {i+1} finding:\n{r}\n" for i, r in enumerate(results)])
        else:
            worker_results_str = "No tools were executed."

        aggregator_prompt_template = get_aggregator_prompt_template()
        aggregator_prompt = aggregator_prompt_template.invoke({
            "user_query": state.get("clarified_query") or user_query,
            "additional_material": additional_material,
            "conversation_history": conversation_history,
            "worker_results": worker_results_str,
            "time_context": state.get("time_context") or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "clarification_block": clarification_block,
        })

        response = await self.aggregator_llm.ainvoke(aggregator_prompt, config=config)
        logger.info("✅ AGGREGATOR STEP COMPLETED")
        logger.info("="*70)

        return {"messages": [response]}
