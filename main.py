
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
            "You are a helpful assistant that can use tools. "
            "You run on-device and should handle requests locally whenever possible. "
            "Only signal cloud handoff when the task is genuinely complex: "
            "for example, when it requires calling multiple tools at once, "
            "when the correct tool is ambiguous given the available options, "
            "or when extracting the right arguments requires multi-step reasoning. "
            "For simple, direct requests that map clearly to a single tool call, "
            "always handle them on-device with full confidence."
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
    Two-phase hybrid routing:
    Phase 1 — Pre-route: count action connectors in the message.
               Multi-action queries go straight to cloud; no FunctionGemma cost wasted.
    Phase 2 — Post-validate: for single-action queries, run FunctionGemma and check
               the output is structurally correct (valid tool name + all required args
               present and non-empty). Deterministic, no probability involved.
    """

    # --- Phase 1: pre-routing by estimated action count ---
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()

    compound_connectors = [" and ", ", and ", " then ", " also ", " as well ", " plus "]
    estimated_actions = 1 + sum(user_text.count(kw) for kw in compound_connectors)

    if estimated_actions >= 2:
        # Multi-action: FunctionGemma 270M can't reliably produce multiple calls.
        # Skip it entirely — go straight to cloud with no wasted inference.
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (pre-routed: multi-action)"
        return cloud

    # --- Phase 2: single-action — run on-device, validate output structure ---
    local = generate_cactus(messages, tools, confidence_threshold=confidence_threshold)
    function_calls = local.get("function_calls", [])

    tool_map = {t["name"]: t["parameters"].get("required", []) for t in tools}
    valid_tool_names = set(tool_map.keys())

    def _is_valid_call(call):
        name = call.get("name", "")
        if name not in valid_tool_names:
            return False  # hallucinated or wrong tool name
        args = call.get("arguments", {})
        # Every required arg must be present and non-empty
        return all(r in args and args[r] not in (None, "", []) for r in tool_map[name])

    structurally_valid = len(function_calls) > 0 and all(_is_valid_call(c) for c in function_calls)

    if structurally_valid:
        local["source"] = "on-device"
        return local

    # Structural check failed — fall back to cloud
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback: invalid output)"
    cloud["local_confidence"] = local.get("confidence", 0)
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
