
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

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
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
    """Multi-signal hybrid routing: run FunctionGemma first, validate result, fall back to cloud if needed."""

    # --- Estimate query complexity from user message ---
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    compound_keywords = [" and ", " also ", " then ", " plus ", " as well ", " additionally "]
    compound_count = sum(user_text.count(kw) for kw in compound_keywords)
    estimated_actions = 1 + compound_count
    num_tools = len(tools)

    # --- Dynamic confidence threshold: stricter for more tools and compound queries ---
    tool_penalty = max(0, num_tools - 1) * 0.02
    compound_penalty = compound_count * 0.05
    dynamic_threshold = min(0.88, confidence_threshold + tool_penalty + compound_penalty)

    # --- Run FunctionGemma on-device ---
    local = generate_cactus(messages, tools)
    confidence = local.get("confidence", 0)
    function_calls = local.get("function_calls", [])

    # Signal 1: confidence below dynamic threshold
    low_confidence = confidence < dynamic_threshold

    # Signal 2: no function calls returned at all
    no_calls = len(function_calls) == 0

    # Signal 3: any call is missing required arguments
    tool_map = {t["name"]: t for t in tools}
    def _valid_args(call):
        tool = tool_map.get(call.get("name", ""))
        if not tool:
            return False
        required = tool["parameters"].get("required", [])
        args = call.get("arguments", {})
        return all(r in args for r in required)
    bad_args = any(not _valid_args(c) for c in function_calls)

    # Signal 4: compound query but fewer calls returned than expected
    insufficient_calls = (estimated_actions >= 2) and (len(function_calls) < estimated_actions)

    # Stay on-device if all signals pass
    if not low_confidence and not no_calls and not bad_args and not insufficient_calls:
        local["source"] = "on-device"
        return local

    # Fall back to cloud
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = confidence
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
