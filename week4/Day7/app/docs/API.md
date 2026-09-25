 API Reference

Base URL: `http://localhost:8000`

## GET /health

Liveness check.

**Response:**
```json
{"status": "ok", "service": "realestate-hub-voice-agent", "version": "1.0.0"}
POST /chat
Send a caller message; get Ayesha's reply.

Request:

json
{
  "message": "Lahore DHA mein 3 bedroom chahiye, 3 crore tak",
  "session_id": "optional-uuid",
  "client_phone": "03001234567"
}
Response:

json
{
  "session_id": "abc-123",
  "reply": "Ji sir, DHA Defence mein 3-bedroom ke 3 options hain...",
  "intent": "buyer",
  "appointment_status": "none",
  "node_trace": ["greeting", "intent_detection", "recommendation"]
}
GET /sessions/{session_id}
Retrieve session state.

Response:

json
{
  "session_id": "abc-123",
  "intent": "buyer",
  "appointment_status": "booked",
  "conversation_length": 6
}
Error Codes
Code	Meaning
400	Bad request (missing message)
404	Session not found
500	Agent error (LLM, RAG, or tool failure)
Rate Limits
Not enforced in demo. Production: 60 requests/min per IP.