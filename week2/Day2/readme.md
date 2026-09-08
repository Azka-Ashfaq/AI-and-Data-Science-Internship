# Week 2 Day 2 - LangChain Agent

## 📋 Overview
LangChain agent with Gemini API, 3 tools (calculator, weather, product price from JSON), memory, structured output, and error handling.

---

## 🚀 Features

| Tool | Description | Data Source |
|------|-------------|-------------|
| `calculator` | Arithmetic operations | In-memory |
| `get_weather` | Weather lookup | FAKE_WEATHER_DB |
| `get_product_price` | Price & specs lookup | **JSON file (NEW)** |

- Reason → Act → Observe loop via `create_agent`
- Memory with InMemorySaver checkpointer
- Structured output with Pydantic
- Error handling with middleware

---

## 📁 Structure
```
├── Lang_cahin.ipynb              # Main notebook
├── Week2_Day2_LangChain_Writeup.docx  # Writeup
├── products.json                  # External data
├── .env                          # API keys
└── README.md
```

---

## 🔧 Setup

```bash
# Install
pip install langchain langchain-google-genai python-dotenv pydantic

# Create .env file
GEMINI_API_KEY=your_key_here

# Run
jupyter notebook Lang_cahin.ipynb
```

---

## 📊 Key Code

**LCEL Pipe Syntax:**
```python
prompt | llm | StrOutputParser()
```

**Tools:**
```python
@tool
def get_product_price(product_name: str) -> str:
    with open("products.json") as f:
        db = json.load(f)
    return f"{product_name}: ${db[key]['price_usd']}"
```

**Memory:**
```python
agent = create_agent(model=llm, tools=TOOLS, checkpointer=InMemorySaver())
thread = {"configurable": {"thread_id": "chat-1"}}
```

**Structured Output:**
```python
class Recommendation(BaseModel):
    product: str
    price: int
    reason: str

agent = create_agent(model=llm, tools=TOOLS, response_format=Recommendation)
```

---

## 🧪 Tests

| Test | Result |
|------|--------|
| Weather Comparison | ✅ Calls tool twice, compares |
| 3-Turn Memory | ✅ Remembers across turns |
| Error Handling | ✅ Middleware catches exceptions |
| Structured Output | ✅ Returns Pydantic model |

---

## 📝 Raw Python vs LangChain

| Aspect | Raw Python | LangChain |
|--------|-----------|-----------|
| Tool Definition | Manual JSON | `@tool` decorator |
| Agent Loop | Manual for-loop | `create_agent` |
| Memory | `scratchpad` dict | `InMemorySaver` |
| Error Handling | try/except | Middleware |

---

## ⚠️ Known Issues

**Gemini Free Tier:** 20 requests/day limit. If you see `429 RESOURCE_EXHAUSTED`, wait until midnight Pacific Time.

---

## 📄 Deliverables

- [x] Jupyter notebook with LangChain agent
- [x] 1-page writeup comparing raw Python vs LangChain
- [x] Annotated reasoning trace
- [x] 3 tools with external data source
- [x] Memory with 3-turn test
- [x] Structured output with Pydantic
- [x] Error handling with middleware

---

## 👤 Author
Azka Ashfaq

AI and Data science intern
