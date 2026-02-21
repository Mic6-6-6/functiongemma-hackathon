"""
Hybrid File Navigation Demo (Cactus x Google DeepMind Hackathon style)

What this shows:
- Local-first routing
- Local model decides which files to read
- Local extraction -> structured JSON + confidence
- Deterministic routing (confidence + reasoning complexity)
- Automatic cloud fallback with FILTERED evidence only

This is a hackathon-friendly scaffold. Replace the stub functions with:
- generate_cactus(...)  -> your on-device FunctionGemma/Cactus call
- generate_cloud(...)   -> your Gemini cloud call
- real file readers/parsers for PDF/XLSX/DOC
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Config
# ----------------------------

SGT = timezone(timedelta(hours=8))
CONFIDENCE_THRESHOLD_HIGH = 0.80
CONFIDENCE_THRESHOLD_LOW = 0.50
TOP_K_CANDIDATES = 3


# ----------------------------
# Example "AI-readable" files (mocked)
# Replace with real parsed outputs from PDF/XLSX/DOC
# ----------------------------

MOCK_FILES = [
    {
        "file_id": "f1",
        "name": "Travel_Itinerary_Feb2026.pdf",
        "type": "pdf",
        "last_modified": "2026-02-19T13:00:00+08:00",
        "chunks": [
            {
                "chunk_id": "c1",
                "page": 1,
                "text": "Booking Confirmation\nPassenger: Peiyao Xu\nFlight: SQ 888\nDeparture: 29 Feb 2026 22:00 (SGT)\nArrival: 01 Mar 2026 00:05 (CST)\nRoute: Singapore (SIN) -> Shanghai (PVG)"
            }
        ],
    },
    {
        "file_id": "f2",
        "name": "old_trip_notes.doc",
        "type": "doc",
        "last_modified": "2025-10-10T08:00:00+08:00",
        "chunks": [
            {
                "chunk_id": "c1",
                "text": "Possible flight idea: 29/02 23:10 SIN to China (not confirmed)"
            }
        ],
    },
    {
        "file_id": "f3",
        "name": "Expense_Log.xlsx",
        "type": "xlsx",
        "last_modified": "2026-02-20T10:30:00+08:00",
        "tables": [
            {
                "table_id": "t1",
                "sheet": "Flights",
                "cells": [
                    ["Type", "Flight"],
                    ["Date", "29 Feb 2026"],
                    ["Dep", "22:00"],
                    ["From", "SIN"],
                    ["To", "PVG"],
                    ["Note", "matches ticket"]
                ],
            }
        ],
    },
]


# ----------------------------
# Data models
# ----------------------------

@dataclass
class CandidateFile:
    file_id: str
    name: str
    file_type: str
    score: float
    reasons: List[str]


@dataclass
class FlightExtraction:
    departure_local: Optional[str] = None
    arrival_local: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    flight_number: Optional[str] = None
    airline: Optional[str] = None
    evidence_chunks: Optional[List[str]] = None
    extraction_confidence: float = 0.0
    consistency_confidence: float = 0.0
    overall_confidence: float = 0.0
    warnings: Optional[List[str]] = None


# ----------------------------
# Model call stubs (replace these)
# ----------------------------

def generate_cactus(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stub for on-device FunctionGemma/Cactus tool-calling."""
    return {
        "model": "FunctionGemma (On-Device Cactus)",
        "raw": "stubbed",
        "messages_seen": len(messages),
        "tools_available": [t["name"] for t in tools],
    }


def generate_cloud(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stub for Gemini cloud fallback."""
    return {
        "model": "Gemini (Cloud)",
        "raw": "stubbed",
        "messages_seen": len(messages),
        "tools_available": [t["name"] for t in tools],
    }


# ----------------------------
# Utility
# ----------------------------

def print_result(title: str, obj: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def now_sgt() -> datetime:
    # Fixed for reproducibility in demo; replace with datetime.now(SGT)
    return datetime(2026, 2, 21, 12, 0, 0, tzinfo=SGT)

def safe_parse_dt(text: str) -> Optional[datetime]:
    # Remove timezone labels in parentheses, e.g. "(SGT)", "(CST)"
    cleaned = re.sub(r"\s*\([A-Za-z]{2,5}\)\s*", " ", text).strip()

    # Try full explicit format first
    m = re.search(r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{1,2}:\d{2})", cleaned)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%d %b %Y %H:%M")
            return dt.replace(tzinfo=SGT)
        except ValueError:
            pass

    # Fallback: DD/MM HH:MM (assume current year for demo)
    m = re.search(r"(\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2})", cleaned)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} 2026", "%d/%m %H:%M %Y")
            return dt.replace(tzinfo=SGT)
        except ValueError:
            pass

    return None


# ----------------------------
# Step 1: Query classify (local)
# ----------------------------

def classify_query_local(user_query: str) -> Dict[str, Any]:
    q = user_query.lower()

    if "next flight" in q or ("flight" in q and "next" in q):
        return {
            "task_type": "flight_lookup",
            "reasoning_complexity": "low",
            "needed_fields": ["departure_local", "origin", "destination"],
            "why": "Direct extraction + nearest future selection"
        }

    # default fallback category
    return {
        "task_type": "general_file_qa",
        "reasoning_complexity": "medium",
        "needed_fields": [],
        "why": "Unclear task; may need broader reasoning"
    }


# ----------------------------
# Step 2: File selection (local AI + deterministic retriever union)
# ----------------------------

def local_ai_propose_files(user_query: str, files: List[Dict[str, Any]]) -> List[CandidateFile]:
    """Mock local AI file selection."""
    results: List[CandidateFile] = []
    q = user_query.lower()

    for f in files:
        score = 0.0
        reasons = []
        name_lower = f["name"].lower()

        if "flight" in q:
            if any(k in name_lower for k in ["itinerary", "travel", "ticket", "boarding", "flight"]):
                score += 0.55
                reasons.append("filename suggests itinerary/flight")
            if f["type"] in {"pdf", "xlsx", "doc"}:
                score += 0.05
                reasons.append("supported AI-readable file type")

        # Light content look
        joined_text = extract_cheap_text(f).lower()
        for kw in ["flight", "departure", "arrival", "sin", "route", "pvg"]:
            if kw in joined_text:
                score += 0.08
                reasons.append(f"content contains '{kw}'")

        # Recency bias
        lm = datetime.fromisoformat(f["last_modified"])
        age_days = (now_sgt() - lm).total_seconds() / 86400.0
        if age_days < 14:
            score += 0.12
            reasons.append("recently modified")
        elif age_days < 90:
            score += 0.06
            reasons.append("moderately recent")

        if score > 0:
            results.append(CandidateFile(
                file_id=f["file_id"],
                name=f["name"],
                file_type=f["type"],
                score=round(min(score, 1.0), 3),
                reasons=reasons
            ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:TOP_K_CANDIDATES]


def deterministic_retriever_union(user_query: str, files: List[Dict[str, Any]], current: List[CandidateFile]) -> List[CandidateFile]:
    current_ids = {c.file_id for c in current}
    extras: List[CandidateFile] = []
    q = user_query.lower()

    if "flight" in q:
        for f in files:
            if f["file_id"] in current_ids:
                continue
            joined = extract_cheap_text(f).lower()
            if any(x in joined for x in ["flight", "departure", "arrival", "sin", "pvg"]):
                extras.append(CandidateFile(
                    file_id=f["file_id"],
                    name=f["name"],
                    file_type=f["type"],
                    score=0.35,
                    reasons=["deterministic keyword fallback candidate"]
                ))

    merged = current + extras
    merged.sort(key=lambda x: x.score, reverse=True)

    # dedupe and keep top K
    out: List[CandidateFile] = []
    seen = set()
    for c in merged:
        if c.file_id in seen:
            continue
        seen.add(c.file_id)
        out.append(c)
        if len(out) >= TOP_K_CANDIDATES:
            break
    return out


def extract_cheap_text(file_obj: Dict[str, Any]) -> str:
    parts = []
    for ch in file_obj.get("chunks", []):
        if "text" in ch:
            parts.append(ch["text"])
    for tb in file_obj.get("tables", []):
        for row in tb.get("cells", []):
            parts.append(" ".join(row))
    return "\n".join(parts)


# ----------------------------
# Step 3: Read + extract (local)
# ----------------------------

def read_files(file_ids: List[str], files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    file_map = {f["file_id"]: f for f in files}
    return [file_map[fid] for fid in file_ids if fid in file_map]


def extract_flight_info_from_files(read_docs: List[Dict[str, Any]]) -> FlightExtraction:
    candidates: List[Tuple[FlightExtraction, datetime]] = []
    warnings: List[str] = []
    conflicts: List[str] = []

    for f in read_docs:
        # Parse text chunks
        for ch in f.get("chunks", []):
            text = ch.get("text", "")

            dep = _extract_dep_datetime(text)
            arr = _extract_arr_datetime(text)
            origin, dest = _extract_route(text)
            flight_no = _extract_flight_number(text)

            if dep or origin or dest:
                fe = FlightExtraction(
                    departure_local=dep.isoformat() if dep else None,
                    arrival_local=arr.isoformat() if arr else None,
                    origin=origin,
                    destination=dest,
                    flight_number=flight_no,
                    airline=(flight_no[:2] if flight_no else None),
                    evidence_chunks=[f"{f['file_id']}:{ch['chunk_id']}"],
                    warnings=[],
                )
                candidates.append((fe, dep if dep else datetime.max.replace(tzinfo=SGT)))

        # Parse table chunks
        for tb in f.get("tables", []):
            dep, arr, origin, dest = _extract_from_table(tb["cells"])
            if dep or origin or dest:
                fe = FlightExtraction(
                    departure_local=dep.isoformat() if dep else None,
                    arrival_local=arr.isoformat() if arr else None,
                    origin=origin,
                    destination=dest,
                    flight_number=None,
                    airline=None,
                    evidence_chunks=[f"{f['file_id']}:{tb['table_id']}"],
                    warnings=[],
                )
                candidates.append((fe, dep if dep else datetime.max.replace(tzinfo=SGT)))

    if not candidates:
        return FlightExtraction(
            warnings=["No flight-like evidence found in selected files."],
            extraction_confidence=0.0,
            consistency_confidence=0.0,
            overall_confidence=0.0,
            evidence_chunks=[]
        )

    # Choose next future flight (strictly > now)
    now = now_sgt()
    future = [(fe, dep) for fe, dep in candidates if dep and dep > now]
    if not future:
        # If none future, choose nearest available but confidence lower
        future = candidates
        warnings.append("No future flight found relative to current time; using nearest available record.")

    future.sort(key=lambda x: x[1])
    best = future[0][0]

    # Consistency checks across candidate records close to best date
    best_dep = datetime.fromisoformat(best.departure_local) if best.departure_local else None
    if best_dep:
        for fe, dep in candidates:
            if dep and abs((dep - best_dep).total_seconds()) < 24 * 3600:
                # compare known fields
                if best.origin and fe.origin and best.origin != fe.origin:
                    conflicts.append(f"Origin conflict: {best.origin} vs {fe.origin}")
                if best.destination and fe.destination and best.destination != fe.destination:
                    conflicts.append(f"Destination conflict: {best.destination} vs {fe.destination}")

    # Score confidence
    extraction_conf, consistency_conf, overall_conf = score_confidence(best, candidates, conflicts)

    # Cross-midnight warning
    if best.departure_local and best.arrival_local:
        dep_dt = datetime.fromisoformat(best.departure_local)
        arr_dt = datetime.fromisoformat(best.arrival_local)
        if arr_dt.date() > dep_dt.date():
            warnings.append("Arrival crosses midnight (arrives on the next calendar day).")

    if conflicts:
        warnings.extend(conflicts)

    best.extraction_confidence = extraction_conf
    best.consistency_confidence = consistency_conf
    best.overall_confidence = overall_conf
    best.warnings = warnings
    return best


def _extract_dep_datetime(text: str) -> Optional[datetime]:
    m = re.search(r"Departure:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        return safe_parse_dt(m.group(1))
    # fallback generic
    return None


def _extract_arr_datetime(text: str) -> Optional[datetime]:
    m = re.search(r"Arrival:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        # We treat parsed timestamp as local for demo simplicity
        return safe_parse_dt(m.group(1))
    return None


def _extract_route(text: str) -> Tuple[Optional[str], Optional[str]]:
    # Example: Route: Singapore (SIN) -> Shanghai (PVG)
    m = re.search(r"Route:\s*(.+?)\s*->\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Fallback airport code style
    m2 = re.search(r"\b(SIN)\b.*?\b(PVG|PEK|PKX|CAN|SZX|SHA)\b", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1), m2.group(2)
    return None, None


def _extract_flight_number(text: str) -> Optional[str]:
    m = re.search(r"Flight:\s*([A-Z]{2}\s?\d{2,4})", text)
    if m:
        return m.group(1).replace(" ", "")
    return None


def _extract_from_table(cells: List[List[str]]) -> Tuple[Optional[datetime], Optional[datetime], Optional[str], Optional[str]]:
    data = {str(row[0]).strip().lower(): str(row[1]).strip() for row in cells if len(row) >= 2}

    dep = None
    arr = None
    origin = None
    dest = None

    date_s = data.get("date")
    dep_s = data.get("dep") or data.get("departure")
    arr_s = data.get("arr") or data.get("arrival")

    if date_s and dep_s:
        try:
            dep = datetime.strptime(f"{date_s} {dep_s}", "%d %b %Y %H:%M").replace(tzinfo=SGT)
        except ValueError:
            pass
    if date_s and arr_s:
        try:
            arr = datetime.strptime(f"{date_s} {arr_s}", "%d %b %Y %H:%M").replace(tzinfo=SGT)
        except ValueError:
            pass

    origin = data.get("from")
    dest = data.get("to")
    return dep, arr, origin, dest


# ----------------------------
# Step 4: Confidence scoring (rule-based + evidence-based)
# ----------------------------

def score_confidence(best: FlightExtraction, all_candidates: List[Tuple[FlightExtraction, datetime]], conflicts: List[str]) -> Tuple[float, float, float]:
    ext = 0.0
    cons = 1.0

    # Extraction evidence points
    if best.departure_local:
        ext += 0.25
    if best.arrival_local:
        ext += 0.15
    if best.origin and best.destination:
        ext += 0.25
    if best.flight_number:
        ext += 0.10
    if best.evidence_chunks:
        ext += 0.10
    if best.departure_local:
        ext += 0.10  # parsed unambiguous date-time
    if len(all_candidates) >= 1:
        ext += 0.05

    ext = min(ext, 1.0)

    # Consistency penalties
    if conflicts:
        cons -= 0.30
    if not best.destination:
        cons -= 0.10
    if not best.origin:
        cons -= 0.10
    cons = max(0.0, min(cons, 1.0))

    overall = round(0.7 * ext + 0.3 * cons, 3)
    return round(ext, 3), round(cons, 3), overall


# ----------------------------
# Step 5: Routing policy
# ----------------------------

def should_fallback(overall_confidence: float, reasoning_complexity: str) -> Tuple[bool, str]:
    if overall_confidence < CONFIDENCE_THRESHOLD_LOW:
        return True, "Low confidence: insufficient or ambiguous extraction."

    if overall_confidence < CONFIDENCE_THRESHOLD_HIGH:
        return True, "Medium confidence: fallback to resolve ambiguity with filtered evidence."

    if reasoning_complexity == "high":
        return True, "High reasoning complexity: local extraction is strong, but cloud may improve final reasoning."

    return False, "No fallback needed: high-confidence extraction and low reasoning complexity."


# ----------------------------
# Step 6: Build response / fallback payload
# ----------------------------

def build_user_answer(ex: FlightExtraction) -> str:
    if not ex.departure_local:
        return "I found flight-related files, but I couldn't confidently extract the departure time."

    dep = datetime.fromisoformat(ex.departure_local)
    dep_str = dep.strftime("%d %b %Y, %-I:%M%p") if hasattr(dep, "strftime") else ex.departure_local
    # Windows may not support %-I, so fallback
    dep_str = dep.strftime("%d %b %Y, %I:%M%p").lstrip("0")

    route = ""
    if ex.origin and ex.destination:
        route = f" from {ex.origin} to {ex.destination}"

    arrival_note = ""
    if ex.arrival_local:
        arr = datetime.fromisoformat(ex.arrival_local)
        arr_str = arr.strftime("%I:%M%p").lstrip("0")
        arrival_note = f", arriving around {arr_str}"

    return f"Your next flight departs on {dep_str} (SGT){route}{arrival_note}."


def build_filtered_fallback_payload(user_query: str, ex: FlightExtraction, read_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence = []
    evidence_refs = set(ex.evidence_chunks or [])

    for f in read_docs:
        for ch in f.get("chunks", []):
            ref = f"{f['file_id']}:{ch['chunk_id']}"
            if ref in evidence_refs:
                evidence.append({"ref": ref, "text": ch.get("text", "")[:1000]})
        for tb in f.get("tables", []):
            ref = f"{f['file_id']}:{tb['table_id']}"
            if ref in evidence_refs:
                evidence.append({"ref": ref, "cells": tb.get("cells", [])})

    return {
        "user_query": user_query,
        "now": now_sgt().isoformat(),
        "local_attempt": {
            "best_guess": {
                "departure_local": ex.departure_local,
                "arrival_local": ex.arrival_local,
                "origin": ex.origin,
                "destination": ex.destination,
                "flight_number": ex.flight_number,
            },
            "confidence": ex.overall_confidence,
            "problems": ex.warnings or [],
        },
        "evidence": evidence[:3],  # filtered evidence only
    }


def format_final_json(answer: str,
                      ex: FlightExtraction,
                      fallback: bool,
                      fallback_reason: str,
                      fallback_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {
        "response": {
            "answer": answer,
            "structured": {
                "departure_local": ex.departure_local,
                "arrival_local": ex.arrival_local,
                "origin": ex.origin,
                "destination": ex.destination,
                "flight_number": ex.flight_number,
                "airline": ex.airline,
            },
            "evidence": ex.evidence_chunks or [],
        },
        "fallback": fallback,
        "confidence": ex.overall_confidence,
        "reasoning_complexity": None,  # filled by caller
        "fallback_reason": fallback_reason,
        "warnings": ex.warnings or [],
    }
    if fallback_payload is not None:
        out["fallback_payload"] = fallback_payload
    return out


# ----------------------------
# Hybrid orchestration
# ----------------------------

def generate_hybrid(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 1) Parse user query
    user_query = messages[-1]["content"]

    # 2) Local classify
    classification = classify_query_local(user_query)
    reasoning_complexity = classification["reasoning_complexity"]

    # 3) Local AI file proposal + deterministic union
    proposed = local_ai_propose_files(user_query, MOCK_FILES)
    candidates = deterministic_retriever_union(user_query, MOCK_FILES, proposed)

    # 4) Read files
    read_docs = read_files([c.file_id for c in candidates], MOCK_FILES)

    # 5) Extract
    ex = extract_flight_info_from_files(read_docs)

    # 6) Route (deterministic; no extra "ask again" loop)
    fallback, fallback_reason = should_fallback(ex.overall_confidence, reasoning_complexity)

    # 7) If fallback, send filtered evidence to cloud (stubbed here)
    fallback_payload = None
    cloud_result = None
    if fallback:
        fallback_payload = build_filtered_fallback_payload(user_query, ex, read_docs)

        # In a real system, call cloud with fallback_payload
        cloud_messages = [
            {"role": "system", "content": "Resolve ambiguity using only provided evidence. Return structured flight answer."},
            {"role": "user", "content": json.dumps(fallback_payload, ensure_ascii=False)}
        ]
        cloud_result = generate_cloud(cloud_messages, tools)

    # 8) Build local answer (or cloud-refined answer later)
    local_answer = build_user_answer(ex)

    final = format_final_json(
        answer=local_answer,
        ex=ex,
        fallback=fallback,
        fallback_reason=fallback_reason,
        fallback_payload=fallback_payload
    )
    final["reasoning_complexity"] = reasoning_complexity
    final["routing_debug"] = {
        "classification": classification,
        "candidate_files": [asdict(c) for c in candidates],
        "cloud_invoked": cloud_result is not None,
        "cloud_result_stub": cloud_result,
    }
    return final


# ----------------------------
# Main (matching your example style)
# ----------------------------


tools = [{
    "name": "read_files",
    "description": "Read AI-readable files (PDF/XLSX/DOC) and return parsed content chunks/tables",
    "parameters": {
        "type": "object",
        "properties": {
            "file_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of files to read",
            }
        },
        "required": ["file_ids"],
    },
}, {
    "name": "fallback_to_cloud",
    "description": "Send filtered evidence and local extraction attempt to cloud for ambiguity resolution",
    "parameters": {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "description": "Filtered evidence package"
            }
        },
        "required": ["payload"],
    },
}]

messages = [
    {"role": "user", "content": "When is my next flight?"}
]

on_device = generate_cactus(messages, tools)
print_result("FunctionGemma (On-Device Cactus)", on_device)

cloud = generate_cloud(messages, tools)
print_result("Gemini (Cloud)", cloud)

hybrid = generate_hybrid(messages, tools)
print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
 
 
