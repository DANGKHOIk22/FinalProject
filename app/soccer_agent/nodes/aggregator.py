import logging
from langgraph.graph.state import RunnableConfig
from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.prompts.agent import get_aggregator_prompt_template

logger = logging.getLogger(__name__)

class AggregatorNode:
    def __init__(self, aggregator_llm):
        self.aggregator_llm = aggregator_llm

    async def aggregator_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """Aggregate results from parallel executions and provide the final response."""
        messages = state.get("messages", [])
        user_query = messages[-1].content if messages else ""
        additional_material_list = state.get("additional_material", [])
        additional_material = ", ".join(additional_material_list) if additional_material_list else "None"
        conversation_history = state.get("conversation_history", "No previous conversation.")
        results = state.get("parallel_results", [])
        
        logger.info("="*70)
        logger.info("🧠 Starting AGGREGATOR STEP")

        # Bypass LLM if only 1 worker result
        if len(results) == 1:
            logger.info("⚡ Single worker result detected. Bypassing aggregator LLM.")
            from langchain_core.messages import AIMessage
            return {
                "messages": [AIMessage(content=results[0])],
            }
            
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
            "time_context": state.get("time_context") or "Unknown"
        })

        response = await self.aggregator_llm.ainvoke(
            aggregator_prompt,
            config=config
        )
        logger.info("✅ AGGREGATOR STEP COMPLETED")
        logger.info("="*70)
            
        return {
            "messages": [response], # Final response from aggregator
        }
