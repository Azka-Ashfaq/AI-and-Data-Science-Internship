# Week 2 Day 1: Agent Foundations

## Overview
A minimal AI agent built from scratch using Python and the Google Gemini API.
Demonstrates the ReAct pattern (Reason → Act → Observe → repeat) with tool calling.

**No frameworks used** — just raw Python to understand what's happening under the hood.

---

## Features
- ✅ ReAct loop (Reason → Act → Observe → repeat)
- ✅ Tool calling with JSON schemas (calculator, weather lookup)
- ✅ Multi-step reasoning (2+ tool calls)
- ✅ Conversation memory + working memory (scratchpad)
- ✅ Logging for debugging
- ✅ max_iterations safeguard (prevents infinite loops)
- ✅ Failure handling (ambiguous requests, tool errors, undefined tools)

---

## Quick Start

### 1. Install dependencies
```bash
pip install google-generativeai python-dotenv
```

### 2. Get a Gemini API key (FREE)
- Go to: https://aistudio.google.com/
- Sign in → click "Get API Key" → copy your key

### 3. Create a `.env` file in the project folder
```
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the agent
```bash
python raw_agent.py
```

> Note: confirm this matches your actual script filename before running — if your file is named differently, use that name here instead.

---

## Tool Schemas

| Tool | Description | Input |
|------|-------------|-------|
| `calculator` | Evaluates arithmetic | `{"expression": "string"}` |
| `get_weather` | Returns weather for a city | `{"city": "string"}` |

---

## Test Results

All tests passed ✅

| Test | Result |
|------|--------|
| Single tool call | ✅ PASSED |
| Multi-step (2 tools) | ✅ PASSED |
| Calculator | ✅ PASSED |
| Ambiguous request | ✅ PASSED |
| Tool error | ✅ PASSED |
| Undefined tool | ✅ PASSED |

---

## Project Structure
```
week2-day1/
├── raw_agent.py                          # Main agent code
├── Agent_Foundations_Writeup_Gemini.docx # Write-up
├── README.md                             # This file
├── .gitignore                            # Excludes secrets
└── .env                                  # API key (NOT shared)
```

---

## Security
- API key stored in `.env` (excluded via `.gitignore`)
- Never commit secrets to version control

---

## Why Frameworks Exist (Task Requirement)
Frameworks like LangChain, LangGraph, and CrewAI exist to turn hand-built loops into production-ready infrastructure — providing standardized tool integrations, memory management, streaming, monitoring, and automatic handling of retries, rate limits, and token counting. Building the loop by hand first makes those abstractions feel like conveniences, not magic.

---

## Author
Azka Ashfaq
AI and Data science Intern
---
