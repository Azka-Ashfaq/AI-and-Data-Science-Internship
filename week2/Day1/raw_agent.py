"""
Week 2 Day 1 — Agent Foundations
Reasoning Loops, Tool Calling & Raw Python Agents (no LangChain/LangGraph)

FREE VERSION using Google Gemini API
--------------------------------------
This version uses Google's Gemini API which is completely free.
API key is loaded from .env file for security.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re
from typing import Dict, List, Any
from datetime import datetime

# =========================================================
# STEP 1: LOAD API KEY FROM .env FILE
# =========================================================
# TO DO: Create a .env file in the project root with:
#   GEMINI_API_KEY=your_api_key_here

load_dotenv()  # Load environment variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("\n⚠️ ERROR: GEMINI_API_KEY not found in .env file!")
    print("   Create a .env file with:")
    print("   GEMINI_API_KEY=your_api_key_here")
    print("\n   Or set as environment variable:")
    print("   Windows: set GEMINI_API_KEY=your_key_here")
    print("   Mac/Linux: export GEMINI_API_KEY=your_key_here")
    exit(1)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Try different model names - find one that works
MODEL_NAMES = [
    "gemini-3.6-flash",   # Latest model
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Find a working model
model = None
for model_name in MODEL_NAMES:
    try:
        test_model = genai.GenerativeModel(model_name)
        # Test if model works
        test_response = test_model.generate_content("Hello")
        model = test_model
        print(f"✅ Using model: {model_name}")
        break
    except Exception as e:
        print(f"   Model {model_name} not available: {e}")

if model is None:
    print("\n⚠️ No model found! Please check your API key.")
    print("   You can also use the mock mode below.")
    # Use mock mode as fallback
    model = None

print("\n" + "="*60)

# =========================================================
# TASK 1: AGENT CONCEPTS & MENTAL MODEL
# =========================================================
"""
AGENT vs CHATBOT vs WORKFLOW:
- Chatbot: Single-turn or multi-turn text response. No tools, no environment interaction.
- Workflow: Fixed, pre-written sequence of steps decided by the developer ahead of time.
- Agent: The LLM decides at runtime which steps to take, in what order, using which tools.

WHAT MAKES SOMETHING "AGENTIC":
  - Autonomy: The model chooses actions, not just generates text
  - Tool Use: Can act on the world and see results
  - Multi-step Planning: Can break a goal into sub-steps dynamically
  - Self-correction: Can notice a bad result and try a different approach

REACT PATTERN (Reason → Act → Observe → repeat):

    +-----------+     +--------+     +---------+
    |  Reason   | --> |  Act   | --> | Observe | --+
    | (thought) |     | (tool) |     | (result)|   |
    +-----------+     +--------+     +---------+   |
          ^                                        |
          +----------------------------------------+
                    loop until final answer

Pseudocode:
    while not done:
        thought = model.think(history)
        if thought.wants_tool:
            result = execute_tool(thought.tool_call)
            history.append(tool_result(result))
        else:
            done = True
            final_answer = thought.text

WHEN AN AGENT IS OVERKILL:
If the task is a single deterministic lookup/calculation, or the steps and order
never change, a plain prompt or a short script is faster, cheaper, and more predictable.
"""

# =========================================================
# TASK 2: TOOL DEFINITIONS (JSON schemas)
# =========================================================
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression (+, -, *, /, parentheses). "
            "Use this whenever the user asks for a numeric calculation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '15 * 7 - 2'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": (
            "Get the current weather (temperature in Celsius and condition) for a given city. "
            "Use this whenever the user asks about weather in a specific city."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city, e.g. 'Lahore' or 'Tokyo'"
                }
            },
            "required": ["city"]
        }
    }
]

# =========================================================
# TOOL EXECUTION
# =========================================================

FAKE_WEATHER_DB = {
    "lahore": {"temp_c": 38, "condition": "Sunny"},
    "karachi": {"temp_c": 33, "condition": "Humid"},
    "islamabad": {"temp_c": 29, "condition": "Cloudy"},
    "london": {"temp_c": 18, "condition": "Rainy"},
    "tokyo": {"temp_c": 27, "condition": "Clear"},
    "new york": {"temp_c": 22, "condition": "Sunny"},
    "sydney": {"temp_c": 18, "condition": "Rainy"},
}

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Runs the requested tool and returns a string result."""
    try:
        if tool_name == "calculator":
            expr = tool_input["expression"]
            allowed_chars = set("0123456789+-*/(). ")
            if not set(expr) <= allowed_chars:
                return f"ERROR: invalid characters in expression '{expr}'"
            result = eval(expr)
            return str(result)

        elif tool_name == "get_weather":
            city = tool_input["city"].strip().lower()
            if city not in FAKE_WEATHER_DB:
                return f"ERROR: no weather data found for city '{tool_input['city']}'"
            data = FAKE_WEATHER_DB[city]
            return f"{data['temp_c']}C, {data['condition']}"

        else:
            return f"ERROR: unknown tool '{tool_name}'"

    except Exception as e:
        return f"ERROR: tool execution failed - {str(e)}"

# =========================================================
# AGENT LOOP WITH GEMINI API (with fallback to mock)
# =========================================================

def build_tools_prompt() -> str:
    """Build the tools description for the prompt."""
    tools_desc = "Available tools:\n"
    for tool in TOOLS:
        tools_desc += f"\n- {tool['name']}: {tool['description']}"
        tools_desc += f"\n  Input schema: {json.dumps(tool['input_schema'], indent=2)}\n"
    return tools_desc

def call_gemini_with_tools(messages: List[Dict]) -> str:
    """Call Gemini with tool schemas in the prompt."""
    
    # If model is None, use mock mode
    if model is None:
        print("   [Using MOCK mode - no API call]")
        return mock_llm_response(messages)
    
    # Build the system prompt with tools
    system_prompt = f"""
You are an AI agent that can use tools to help users.

{build_tools_prompt()}

IMPORTANT INSTRUCTIONS:
1. If you need to use a tool, respond with exactly this format:
   TOOL_CALL: tool_name
   INPUT: {{"param1": "value1", "param2": "value2"}}

2. If you have the final answer, respond with:
   FINAL: your answer here

3. You can use multiple tools if needed. After receiving a tool result,
   analyze it and decide next action.

Current conversation history:
{json.dumps(messages, indent=2)}

Now respond:
"""
    
    try:
        response = model.generate_content(system_prompt)
        return response.text
    except Exception as e:
        print(f"   [API Error: {e}]")
        # Fallback to mock mode
        return mock_llm_response(messages)

def mock_llm_response(messages: List[Dict]) -> str:
    """Mock response when API is not available."""
    # Extract user query
    user_query = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
    
    user_lower = user_query.lower()
    
    # Check if there are tool results to process
    tool_results = []
    for msg in messages:
        if isinstance(msg.get("content"), str) and "Tool result:" in msg.get("content", ""):
            tool_results.append(msg["content"])
    
    # If we have tool results, provide final answer
    if tool_results:
        # Check for weather results
        temps = []
        for r in tool_results:
            match = re.search(r'(\d+)C', r)
            if match:
                temps.append(int(match.group(1)))
        
        if len(temps) >= 2:
            return f"FINAL: Comparing the results: the warmer city had {max(temps)}C. Based on the lookups, that's the warmer of the two."
        elif len(temps) == 1:
            return f"FINAL: The current temperature is {temps[0]}C."
        elif any("ERROR" in r for r in tool_results):
            return "FINAL: There was an error with the tool call. Please check the city name and try again."
        else:
            return "FINAL: Tool execution completed."
    
    # Weather requests
    if "weather" in user_lower:
        cities = ["lahore", "tokyo", "london", "new york", "sydney", "karachi", "islamabad"]
        found_cities = [c for c in cities if c in user_lower]
        if found_cities:
            return f"TOOL_CALL: get_weather\nINPUT: {{\"city\": \"{found_cities[0]}\"}}"
        else:
            return f"FINAL: Could you tell me which city you're asking about?"
    
    # Calculator requests
    if "calculate" in user_lower or any(op in user_lower for op in ["+", "-", "*", "/"]):
        expr_match = re.search(r'[\d\s\+\-\*/\(\)\.]+', user_query)
        if expr_match:
            expr = expr_match.group().strip()
            # Remove "calculate" and "for me" from expression
            expr = re.sub(r'calculate\s*', '', expr, flags=re.IGNORECASE)
            expr = re.sub(r'\s*for me\s*', '', expr, flags=re.IGNORECASE)
            return f"TOOL_CALL: calculator\nINPUT: {{\"expression\": \"{expr}\"}}"
    
    # Email requests (undefined tool)
    if any(word in user_lower for word in ["email", "send", "message", "sms", "mail"]):
        return "FINAL: I don't have a tool for sending emails or messages. I only have a calculator and weather lookup available."
    
    # Ambiguous requests
    if "nice outside" in user_lower or "good weather" in user_lower:
        return "FINAL: Could you tell me which city you're asking about? I need a location to check the weather."
    
    return "FINAL: I don't have enough information or the right tool to complete this request."

def parse_response(response_text: str) -> Dict[str, Any]:
    """Parse the model's response to find tool calls or final answer."""
    
    # Check for tool call
    if "TOOL_CALL:" in response_text:
        lines = response_text.strip().split('\n')
        tool_name = None
        tool_input = {}
        reasoning = []
        
        for line in lines:
            if line.startswith("TOOL_CALL:"):
                tool_name = line.replace("TOOL_CALL:", "").strip()
            elif line.startswith("INPUT:"):
                try:
                    tool_input = json.loads(line.replace("INPUT:", "").strip())
                except:
                    match = re.search(r'INPUT:\s*({.*?})', line, re.DOTALL)
                    if match:
                        try:
                            tool_input = json.loads(match.group(1))
                        except:
                            tool_input = {}
            else:
                if line.strip() and not line.startswith("FINAL:"):
                    reasoning.append(line.strip())
        
        if tool_name:
            return {
                "type": "tool_use",
                "name": tool_name,
                "input": tool_input,
                "id": f"call_{hash(str(tool_name) + str(tool_input))}",
                "reasoning": "\n".join(reasoning)
            }
    
    # Check for final answer
    if "FINAL:" in response_text:
        final_text = response_text.split("FINAL:")[-1].strip()
        return {
            "type": "text",
            "text": final_text
        }
    
    # Default: treat as text
    return {
        "type": "text",
        "text": response_text.strip()
    }

# =========================================================
# TASK 3 + 4: FULL AGENT LOOP with logging
# =========================================================

def run_agent(user_task: str, max_iterations: int = 6):
    """Run the agent loop for a given task."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING AGENT")
    print(f"📝 Task: {user_task}")
    print(f"{'='*60}\n")
    
    messages = [{"role": "user", "content": user_task}]
    scratchpad = {"tool_calls_made": []}  # working memory
    
    for step in range(1, max_iterations + 1):
        print(f"--- Step {step}/{max_iterations} ---")
        
        # 🔄 Step 1: Call the model (Reason)
        response_text = call_gemini_with_tools(messages)
        parsed = parse_response(response_text)
        
        # Log the reasoning
        if parsed.get("reasoning"):
            print(f"[Reasoning] {parsed['reasoning']}")
        else:
            # Clean up response text for logging
            clean_text = response_text[:200] + "..." if len(response_text) > 200 else response_text
            # Remove FINAL: or TOOL_CALL: for cleaner output
            clean_text = re.sub(r'FINAL:\s*', '', clean_text)
            clean_text = re.sub(r'TOOL_CALL:\s*', '', clean_text)
            print(f"[Reasoning] {clean_text}")
        
        # 🔄 Step 2: Check for final answer
        if parsed["type"] == "text":
            print(f"\n✅ [Final answer] {parsed['text']}")
            scratchpad["status"] = "complete"
            return parsed['text']
        
        # 🔄 Step 3: Execute tool (Act)
        if parsed["type"] == "tool_use":
            tool_name = parsed["name"]
            tool_input = parsed["input"]
            
            print(f"🔧 [Tool call] {tool_name}({json.dumps(tool_input)})")
            
            # Execute the tool
            result = execute_tool(tool_name, tool_input)
            print(f"📊 [Observation] {result}")
            
            # Update working memory (scratchpad)
            scratchpad["tool_calls_made"].append({
                "tool": tool_name,
                "input": tool_input,
                "result": result
            })
            
            # 🔄 Step 4: Append tool result (Observe)
            messages.append({
                "role": "assistant",
                "content": f"Tool result: {result}"
            })
            
            print(f"📝 [Working Memory] {len(scratchpad['tool_calls_made'])} tool calls made")
        
        print()
    
    # ⚠️ Guardrail: max_iterations reached
    print(f"\n⚠️ [Guardrail] max_iterations ({max_iterations}) reached without a final answer.")
    scratchpad["status"] = "incomplete"
    return None

# =========================================================
# TASK 2: SINGLE TOOL CALL DEMO
# =========================================================

def single_tool_call_demo():
    """Task 2: Single request where model chooses a tool."""
    print(f"\n{'='*60}")
    print("TASK 2: Single Tool Call Demo")
    print(f"{'='*60}\n")
    
    user_task = "What's the weather in Lahore right now?"
    print(f"📝 Query: {user_task}\n")
    
    messages = [{"role": "user", "content": user_task}]
    
    # First call - get tool choice
    response_text = call_gemini_with_tools(messages)
    parsed = parse_response(response_text)
    
    if parsed["type"] == "tool_use":
        print(f"🔧 [Model requested tool] {parsed['name']}")
        print(f"📥 [Input] {json.dumps(parsed['input'])}")
        
        result = execute_tool(parsed["name"], parsed["input"])
        print(f"📤 [Tool result] {result}")
        
        # Send back tool result and get final answer
        messages.append({"role": "assistant", "content": f"Tool result: {result}"})
        
        final_response = call_gemini_with_tools(messages)
        final_parsed = parse_response(final_response)
        
        if final_parsed["type"] == "text":
            print(f"✅ [Final answer] {final_parsed['text']}")
    else:
        print(f"Model answered: {parsed.get('text', 'No response')}")

# =========================================================
# TASK 5: BREAK THE AGENT (Failure Modes)
# =========================================================

def break_agent_demo():
    """Task 5: Deliberately break the agent with different scenarios."""
    print(f"\n{'='*60}")
    print("TASK 5: Failure Mode Tests")
    print(f"{'='*60}\n")
    
    # Test 1: Ambiguous request
    print("🔴 TEST 1: Ambiguous Request")
    print("   Query: 'Is it nice outside?'")
    print("   Expected: Agent asks clarifying question\n")
    result = run_agent("Is it nice outside?", max_iterations=3)
    print(f"Result: {result}\n")
    
    # Test 2: Tool that will error
    print("🔴 TEST 2: Tool Error")
    print("   Query: 'What's the weather in Atlantis?'")
    print("   Expected: Tool returns ERROR string\n")
    result = run_agent("What's the weather in Atlantis?", max_iterations=3)
    print(f"Result: {result}\n")
    
    # Test 3: Undefined tool request
    print("🔴 TEST 3: Undefined Tool")
    print("   Query: 'Send an email to my professor'")
    print("   Expected: Agent declines gracefully\n")
    result = run_agent("Send an email to my professor about tomorrow's class.", max_iterations=3)
    print(f"Result: {result}\n")

# =========================================================
# FAILURE MODES & MITIGATIONS
# =========================================================
"""
CONCRETE FAILURE MODES AND MITIGATIONS:

1. INFINITE LOOP
   - Symptom: Agent never completes, keeps calling tools
   - Mitigation: max_iterations safeguard

2. AMBIGUOUS REQUEST
   - Symptom: "Is it nice outside?" no city/metric given
   - Mitigation: Ask clarifying question before calling tools

3. HALLUCINATED TOOL CALLS
   - Symptom: Model invokes non-existent tools
   - Mitigation: Validate tool names against registry

4. WRONG TOOL ARGUMENTS
   - Symptom: Tool called with incorrect parameter types
   - Mitigation: JSON schema validation

5. TOOL RETURNS ERROR
   - Symptom: get_weather asked for city not in database
   - Mitigation: Return clear ERROR string

6. SILENT ERRORS
   - Symptom: Tool fails internally but agent continues
   - Mitigation: Explicit error handling with user-friendly messages
"""

# =========================================================
# WHY FRAMEWORKS EXIST
# =========================================================
"""
WHY FRAMEWORKS LIKE LANGCHAIN/LANGGRAPH/CREWAI EXIST:

While building a raw agent from scratch is educational, frameworks provide
production-ready infrastructure. They offer standardized tool integrations,
automatic prompt engineering, built-in memory management, streaming,
monitoring dashboards, and advanced patterns like multi-agent collaboration.
Frameworks also handle edge cases (retries, rate limiting, token counting)
and provide abstractions that make it easier to scale from prototypes to
production applications with proper observability and testing.
"""

# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏗️  WEEK 2 DAY 1: AGENT FOUNDATIONS")
    print("Raw Python Agent with Gemini API (Free)")
    print("="*60)
    
    # Run all tasks
    try:
        # TASK 2: Single tool call demo
        single_tool_call_demo()
        
        print("\n" + "="*60)
        print("TASK 3: Multi-Step Task (2+ Tool Calls)")
        print("="*60)
        
        # TASK 3: Multi-step weather comparison
        result = run_agent("Look up the weather in Lahore and Tokyo and tell me which is warmer.")
        
        print("\n" + "="*60)
        print("TASK 3: Calculator Test")
        print("="*60)
        
        # TASK 3: Calculator test
        result = run_agent("Calculate 45 * 12 + 7 for me")
        
        # TASK 5: Break the agent
        break_agent_demo()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETE!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()