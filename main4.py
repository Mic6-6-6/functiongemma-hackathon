import sys
sys.path.insert(0, "cactus/python/src")

import json
from main import generate_cactus
from benchmark import BENCHMARKS


def run_case(case):
    result = generate_cactus(case["messages"], case["tools"])

    print(f"\n{'='*60}")
    print(f"  [{case['difficulty'].upper()}] {case['name']}")
    print(f"{'='*60}")
    print(f"  User     : {case['messages'][0]['content']}")
    print(f"  Tools    : {[t['name'] for t in case['tools']]}")
    print(f"  Expected : {json.dumps(case['expected_calls'])}")
    print(f"  Got      : {json.dumps(result.get('function_calls', []))}")
    print(f"  Confidence : {result.get('confidence', 0):.4f}")
    print(f"  Time       : {result.get('total_time_ms', 0):.0f}ms")
    if not result.get('function_calls') and (result.get('response') or result.get('error')):
        print(f"  Raw response: {result.get('response') or result.get('error')}")



total = len(BENCHMARKS)
for i, case in enumerate(BENCHMARKS, 1):
    print(f"\n[{i}/{total}]", end="")
    run_case(case)
