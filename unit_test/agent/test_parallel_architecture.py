import asyncio
import logging
import sys
# Configure logging to stdout
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from app.soccer_agent.agent import SoccerAgent
from app.schema.chat import ChatRequest

async def run_test():
    print("Testing Parallel Execution...")
    agent = SoccerAgent()
    req = ChatRequest(
        user_query="What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley, and who is kylian mbappé?", 
        user_id="test_runner"
    )
    res = await agent.run(req)
    print("\n" + "="*50)
    print("FINAL RESPONSE:")
    print("="*50)
    print(res)
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_test())
