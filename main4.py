
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json
from cactus import cactus_init, cactus_complete, cactus_destroy


TOOL_GET_WEATHER = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"],
    },
}

TOOL_SET_ALARM = {
    "name": "set_alarm",
    "description": "Set an alarm for a given time",
    "parameters": {
        "type": "object",
        "properties": {
            "hour":   {"type": "integer", "description": "Hour to set the alarm for"},
            "minute": {"type": "integer", "description": "Minute to set the alarm for"},
        },
        "required": ["hour", "minute"],
    },
}

TOOL_SEND_MESSAGE = {
    "name": "send_message",
    "description": "Send a message to a contact",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Name of the person to send the message to"},
            "message":   {"type": "string", "description": "The message content to send"},
        },
        "required": ["recipient", "message"],
    },
}

# Pick a few representative cases to inspect
CASES = [
    {
        "label": "Easy — weather (1 tool)",
        "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
        "tools": [TOOL_GET_WEATHER],
        "expected": [{"name": "get_weather", "arguments": {"location": "San Francisco"}}],
    },
    {
        "label": "Easy — alarm (integers, implicit minute=0)",
        "messages": [{"role": "user", "content": "Set an alarm for 10 AM."}],
        "tools": [TOOL_SET_ALARM],
        "expected": [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}],
    },
    {
        "label": "Easy — send message (string extraction)",
        "messages": [{"role": "user", "content": "Send a message to Alice saying good morning."}],
        "tools": [TOOL_SEND_MESSAGE],
        "expected": [{"name": "send_message", "arguments": {"recipient": "Alice", "message": "good morning"}}],
    },
    {
        "label": "Medium — weather among 4 tools (tool selection)",
        "messages": [{"role": "user", "content": "What's the weather in Berlin?"}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_GET_WEATHER],
        "expected": [{"name": "get_weather", "arguments": {"location": "Berlin"}}],
    },
]


def run_case(case):
    model = cactus_init(functiongemma_path)

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + case["messages"],
        tools=case["tools"],
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        confidence_threshold=0.0,
    )

    cactus_destroy(model)

    print(f"\n{'='*60}")
    print(f"  {case['label']}")
    print(f"{'='*60}")
    print(f"  User   : {case['messages'][0]['content']}")
    print(f"  Tools  : {[t['name'] for t in case['tools']]}")
    print(f"\n--- Raw output from cactus_complete ---")
    print(raw_str)

    try:
        parsed = json.loads(raw_str)
        print(f"\n--- Parsed fields ---")
        print(f"  confidence    : {parsed.get('confidence')}")
        print(f"  cloud_handoff : {parsed.get('cloud_handoff')}")
        print(f"  function_calls: {json.dumps(parsed.get('function_calls', []), indent=4)}")
    except json.JSONDecodeError as e:
        print(f"\n  [JSON parse error: {e}]")

    print(f"\n--- Expected ---")
    print(f"  {json.dumps(case['expected'], indent=4)}")


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
