
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    system_prompt = (
        "You are a device assistant. Call tools with arguments extracted directly from the user's message — all required information is already there, never ask for it. "
        "Recognise colloquial intent: 'wake me up' → set_alarm, 'remind me' → create_reminder, 'text/message someone' → send_message. "
        "For compound requests (and/then/also/plus): rank tools by relevance to each action, then call the top N tools where N = number_of_connectors + 1. "
        "Numeric arguments are always positive and extracted from the message (e.g. '5 minute timer' → minutes: 5, never -5). No text responses."
    )

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=0,
        confidence_threshold=0.0,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
        "response": raw.get("response"),
        "error": raw.get("error"),
    }



def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.6):
    """
    Two-step agent routing:

    Step 1 — Triage: Run FunctionGemma, coerce types, and score the output with:
        overall_confidence = 0.40 * raw_conf        # model generation certainty
                           + 0.30 * arg_validity    # fraction of calls with all required args filled
                           + 0.20 * arg_coverage    # fraction of string arg values found in user message
                           + 0.10 * call_completeness # enough calls for estimated actions
    Decision: keep on-device if overall_confidence >= threshold, else fall back.

    Step 2 — Route: return on-device result, or hand off to cloud.
    """

    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    compound_connectors = [" and ", ", and ", " then ", " also ", " as well ", " plus "]
    compound_count = sum(user_text.count(kw) for kw in compound_connectors)
    estimated_actions = 1 + compound_count

    # ── Step 1: Triage ────────────────────────────────────────────────────────

    local = generate_cactus(messages, tools)
    raw_conf = local.get("confidence", 0.0)

    # Coerce numeric argument types (model often returns "5" instead of 5)
    tool_props = {t["name"]: t["parameters"].get("properties", {}) for t in tools}

    def _coerce(call):
        name = call.get("name", "")
        args = dict(call.get("arguments", {}))
        for key, schema in tool_props.get(name, {}).items():
            if key not in args:
                continue
            val, typ = args[key], schema.get("type", "string")
            if typ == "integer" and not isinstance(val, int):
                try:
                    args[key] = int(val)
                except (ValueError, TypeError):
                    pass
            elif typ == "number" and not isinstance(val, (int, float)):
                try:
                    args[key] = float(val)
                except (ValueError, TypeError):
                    pass
        return {**call, "arguments": args}

    function_calls = [_coerce(c) for c in local.get("function_calls", [])]
    local["function_calls"] = function_calls

    tool_map = {t["name"]: t["parameters"].get("required", []) for t in tools}
    valid_tool_names = set(tool_map.keys())

    def _valid_call(call):
        name = call.get("name", "")
        if name not in valid_tool_names:
            return False
        args = call.get("arguments", {})
        return all(r in args and args[r] not in (None, "", []) for r in tool_map[name])

    def _arg_coverage(call):
        string_vals = [v.lower() for v in call.get("arguments", {}).values() if isinstance(v, str) and v]
        if not string_vals:
            return 1.0
        return sum(1 for v in string_vals if v in user_text) / len(string_vals)

    if not function_calls:
        triage = "cloud"
        reason = "no calls"
        overall_confidence = 0.0
    else:
        arg_validity     = sum(1 for c in function_calls if _valid_call(c)) / len(function_calls)
        arg_coverage     = sum(_arg_coverage(c) for c in function_calls) / len(function_calls)
        call_completeness = min(1.0, len(function_calls) / max(1, estimated_actions))
        overall_confidence = max(0.0, min(1.0,
            0.40 * raw_conf         +
            0.30 * arg_validity     +
            0.20 * arg_coverage     +
            0.10 * call_completeness
        ))
        triage = "on-device" if overall_confidence >= confidence_threshold else "cloud"
        reason = "low confidence" if triage == "cloud" else None

    local["overall_confidence"] = overall_confidence

    # ── Step 2: Route ─────────────────────────────────────────────────────────

    if triage == "on-device":
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = f"cloud (fallback: {reason})"
    cloud["local_confidence"] = raw_conf
    cloud["overall_confidence"] = overall_confidence
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)