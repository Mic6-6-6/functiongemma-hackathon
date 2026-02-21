
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools, confidence_threshold=0.7):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": (
            "Call the best matching tool using values taken directly from the user's message. "
            "String args: copy the user's exact words. "
            "Integer args: plain numbers only, never strings (e.g. hour=10, minute=0, minutes=5). "
            "Times: '6 AM'→hour=6,minute=0; '8:15 AM'→hour=8,minute=15. "
            "Always include every required argument."
        )}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        confidence_threshold=confidence_threshold,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "cloud_handoff": True,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
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
    Formula-based hybrid routing.

    Runs FunctionGemma unconditionally (confidence_threshold=0 disables its built-in
    early-exit so we always get a full output to evaluate). Then computes an
    overall_confidence from five weighted components:

        overall_confidence = 0.40 * raw_conf          # model's own score
                           + 0.20 * simplicity        # penalise compound queries
                           + 0.15 * tool_clarity      # penalise large tool sets
                           + 0.15 * call_completeness # got enough calls for the query
                           + 0.10 * arg_validity      # all required args present & non-empty

    Falls back to cloud if overall_confidence < confidence_threshold.
    """

    # --- Query analysis ---
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    compound_connectors = [" and ", ", and ", " then ", " also ", " as well ", " plus "]
    compound_count = sum(user_text.count(kw) for kw in compound_connectors)
    estimated_actions = 1 + compound_count
    tool_count = len(tools)

    # --- Run FunctionGemma — disable built-in handoff so we always get a full output ---
    local = generate_cactus(messages, tools, confidence_threshold=0.0)
    raw_conf = local.get("confidence", 0.0)
    function_calls = local.get("function_calls", [])

    # --- Structural arg check ---
    tool_map = {t["name"]: t["parameters"].get("required", []) for t in tools}
    valid_tool_names = set(tool_map.keys())

    def _valid_call(call):
        name = call.get("name", "")
        if name not in valid_tool_names:
            return False
        args = call.get("arguments", {})
        return all(r in args and args[r] not in (None, "", []) for r in tool_map[name])

    all_args_valid = len(function_calls) > 0 and all(_valid_call(c) for c in function_calls)

    # --- Five-component confidence formula ---

    # 1. Raw model confidence (strongest direct signal)
    w_raw = 0.40

    # 2. Query simplicity: each compound connector halves the simplicity
    simplicity = max(0.0, 1.0 - compound_count * 0.5)
    w_simplicity = 0.20

    # 3. Tool clarity: more tools = more selection ambiguity
    tool_clarity = 1.0 / (1.0 + 0.2 * max(0, tool_count - 1))
    w_tool_clarity = 0.15

    # 4. Call completeness: returned calls vs estimated actions needed
    call_completeness = min(1.0, len(function_calls) / max(1, estimated_actions))
    w_call_completeness = 0.15

    # 5. Argument validity: binary — all required args present and non-empty
    arg_validity = 1.0 if all_args_valid else 0.0
    w_arg_validity = 0.10

    overall_confidence = (
        w_raw             * raw_conf         +
        w_simplicity      * simplicity        +
        w_tool_clarity    * tool_clarity      +
        w_call_completeness * call_completeness +
        w_arg_validity    * arg_validity
    )
    overall_confidence = max(0.0, min(1.0, overall_confidence))

    local["overall_confidence"] = overall_confidence

    if overall_confidence >= confidence_threshold:
        local["source"] = "on-device"
        return local

    # Overall confidence too low — fall back to cloud
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
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
