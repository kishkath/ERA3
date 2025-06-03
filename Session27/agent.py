# agent.py

import asyncio
import yaml
from core.loop import AgentLoop
from core.session import MultiMCP
from core.context import MemoryItem, AgentContext
import datetime
from pathlib import Path
import json
import re
from modules.heuristics import QueryHeuristics
from modules.conversation_index import ConversationIndex

def log(stage: str, msg: str):
    """Simple timestamped console logger."""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {msg}")

async def main():
    print("🧠 Cortex-R Agent Ready")
    current_session = None
    
    # Initialize heuristics and conversation index
    heuristics = QueryHeuristics()
    conversation_index = ConversationIndex()

    with open("config/profiles.yaml", "r") as f:
        profile = yaml.safe_load(f)
        mcp_servers_list = profile.get("mcp_servers", [])
        mcp_servers = {server["id"]: server for server in mcp_servers_list}

    multi_mcp = MultiMCP(server_configs=list(mcp_servers.values()))
    await multi_mcp.initialize()

    try:
        while True:
            user_input = input("🧑 What do you want to solve today? → ")
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'new':
                current_session = None
                continue
            if user_input.lower() == 'history':
                if current_session:
                    history = conversation_index.get_session_history(current_session)
                    print("\n📜 Session History:")
                    for entry in history:
                        print(f"\nQ: {entry['query']}")
                        print(f"A: {entry['response']}")
                        print(f"Time: {datetime.datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                continue

            # Apply heuristics to the query
            heuristic_result = heuristics.apply_heuristics(user_input, "")
            if heuristic_result.applied_heuristics:
                log("heuristics", f"Applied heuristics: {', '.join(heuristic_result.applied_heuristics)}")
                if heuristic_result.confidence_score < 0.7:
                    log("heuristics", f"Warning: Low confidence score ({heuristic_result.confidence_score:.2f})")
                user_input = heuristic_result.modified_query

            # Search for similar past conversations
            similar_conversations = conversation_index.search_similar_conversations(user_input)
            if similar_conversations:
                log("memory", f"Found {len(similar_conversations)} similar past conversations")
                if similar_conversations[0]['similarity_score'] > 0.8:
                    print("\n💡 Similar past conversation found:")
                    print(f"Q: {similar_conversations[0]['query']}")
                    print(f"A: {similar_conversations[0]['response']}")

            while True:
                context = AgentContext(
                    user_input=user_input,
                    session_id=current_session,
                    dispatcher=multi_mcp,
                    mcp_server_descriptions=mcp_servers,
                )
                agent = AgentLoop(context)
                if not current_session:
                    current_session = context.session_id

                result = await agent.run()

                if isinstance(result, dict):
                    answer = result["result"]
                    
                    # Apply heuristics to the result
                    result_heuristic = heuristics.apply_heuristics(user_input, answer)
                    if result_heuristic.applied_heuristics:
                        log("heuristics", f"Applied result heuristics: {', '.join(result_heuristic.applied_heuristics)}")
                        answer = result_heuristic.modified_result
                    
                    # Index the conversation
                    conversation_index.add_conversation(
                        session_id=current_session,
                        query=user_input,
                        response=answer,
                        metadata={
                            "heuristics_applied": heuristic_result.applied_heuristics,
                            "confidence_score": heuristic_result.confidence_score
                        }
                    )
                    
                    if "FINAL_ANSWER:" in answer:
                        print(f"\n💡 Final Answer: {answer.split('FINAL_ANSWER:')[1].strip()}")
                        break
                    elif "FURTHER_PROCESSING_REQUIRED:" in answer:
                        user_input = answer.split("FURTHER_PROCESSING_REQUIRED:")[1].strip()
                        print(f"\n🔁 Further Processing Required: {user_input}")
                        continue  # 🧠 Re-run agent with updated input
                    else:
                        print(f"\n💡 Final Answer (raw): {answer}")
                        break
                else:
                    print(f"\n💡 Final Answer (unexpected): {result}")
                    break
    except KeyboardInterrupt:
        print("\n👋 Received exit signal. Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())



# Find the ASCII values of characters in INDIA and then return sum of exponentials of those values.
# How much Anmol singh paid for his DLF apartment via Capbridge? 
# What do you know about Don Tapscott and Anthony Williams?
# What is the relationship between Gensol and Go-Auto?
# which course are we teaching on Canvas LMS? "H:\DownloadsH\How to use Canvas LMS.pdf"
# Summarize this page: https://theschoolof.ai/
# What is the log value of the amount that Anmol singh paid for his DLF apartment via Capbridge? 