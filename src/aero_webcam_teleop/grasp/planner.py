"""
LLM grasp planner (Claude via the Anthropic API, optional).

Given a detected object, asks Claude for a grasp type and force level from
the pose library. Calls run in a background thread so the control loop never
blocks. Without the ``anthropic`` package or an ``ANTHROPIC_API_KEY`` (env or
``.env``), it answers from the fallback table in ``poses.py``.

* ``request_plan`` returns whether the request was accepted, so the caller
  can recover when the planner is busy or cooling down.
* The in-flight flag is guarded by a lock.
* JSON parsing extracts the first ``{...}`` block instead of relying on how
  markdown fences are laid out.
* The model is configurable (``model=`` or ``AERO_GRASP_MODEL``).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .poses import FORCE_LEVELS, GRASP_TYPES, default_for_object, normalize_label

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

try:
    import anthropic  # type: ignore
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

DEFAULT_MODEL = os.environ.get("AERO_GRASP_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are the grasp planner for the TetherIA Aero Hand Open, a five-finger \
anthropomorphic robotic hand with 16 joints driven by 7 actuators.

Given an object description, choose a grasp type and force level. Reply with a single JSON \
object and nothing else.

Grasp types:
- "power": full-hand wrap. Cylinders, large spheres, large prisms.
- "pinch": index tip + thumb tip. Small objects, pens, coins.
- "tripod": thumb + index + middle tips. Cubes, medium objects.
- "lateral_pinch": thumb against the side of the index. Flat or thin objects.
- "palmar": object rests in the palm, all fingers close. Soft or squishy objects.
- "open": fully open. Only for approach or release.

Force levels:
- "light": soft, fragile or compliant objects.
- "medium": everyday rigid objects (wood, plastic).
- "firm": heavy, slippery or dense objects (metal, glass).

Format:
{"grasp_type": "<type>", "force_level": "<level>", "reasoning": "<one sentence>"}"""


@dataclass
class GraspPlan:
    object_label: str
    grasp_type: str
    force_level: str
    reasoning: str
    source: str  # "llm" | "fallback"
    latency_ms: float = 0.0


def fallback_plan(object_label: str, note: str = "") -> GraspPlan:
    grasp, force, reason = default_for_object(object_label)
    return GraspPlan(normalize_label(object_label), grasp, force, reason + note, "fallback")


def build_prompt(object_label: str, size_cm: Optional[float], depth_m: Optional[float],
                 confidence: float) -> str:
    lines = [f"Object: {object_label.replace('_', ' ')}",
             f"Detection confidence: {confidence:.0%}"]
    if size_cm is not None:
        lines.append(f"Estimated size: {size_cm:.1f} cm")
    if depth_m is not None:
        lines.append(f"Distance from camera: {depth_m:.2f} m")
    lines.append("Choose the grasp type and force level for picking up this object.")
    return "\n".join(lines)


def parse_reply(text: str) -> dict:
    """Extract and validate the JSON object from a model reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in reply: {text[:80]!r}")
    data = json.loads(match.group(0))
    grasp = str(data.get("grasp_type", "")).strip().lower()
    force = str(data.get("force_level", "")).strip().lower()
    if grasp not in GRASP_TYPES:
        raise ValueError(f"Invalid grasp_type {grasp!r}")
    if force not in FORCE_LEVELS:
        raise ValueError(f"Invalid force_level {force!r}")
    return {"grasp_type": grasp, "force_level": force,
            "reasoning": str(data.get("reasoning", "")).strip()}


class GraspPlanner:
    """
    Parameters
    ----------
    on_plan    : callback(GraspPlan), called from a worker thread
    cooldown_s : minimum seconds between requests
    model      : Anthropic model id
    client     : inject an Anthropic-like client (used by tests)
    """

    def __init__(
        self,
        on_plan: Optional[Callable[[GraspPlan], None]] = None,
        cooldown_s: float = 2.0,
        model: str = DEFAULT_MODEL,
        client=None,
    ):
        self.on_plan = on_plan
        self.cooldown_s = cooldown_s
        self.model = model
        self._lock = threading.Lock()
        self._pending = False
        self._last_request_t = -float("inf")

        self._client = client
        if self._client is None and ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            self._client = anthropic.Anthropic()
        if self._client is None:
            print("[planner] No Anthropic client (package or ANTHROPIC_API_KEY missing); "
                  "using fallback grasp table.")

    @property
    def uses_llm(self) -> bool:
        return self._client is not None

    @property
    def is_pending(self) -> bool:
        return self._pending

    def plan(self, object_label: str, size_cm: Optional[float] = None,
             depth_m: Optional[float] = None, confidence: float = 1.0) -> GraspPlan:
        """Synchronous plan: LLM if available, fallback on any failure."""
        if self._client is None:
            return fallback_plan(object_label)
        t0 = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user",
                           "content": build_prompt(object_label, size_cm, depth_m, confidence)}],
            )
            text = "".join(getattr(b, "text", "") for b in response.content)
            data = parse_reply(text)
        except Exception as exc:
            print(f"[planner] LLM call failed ({exc}); using fallback.")
            return fallback_plan(object_label, " (LLM unavailable)")
        return GraspPlan(
            object_label=normalize_label(object_label),
            latency_ms=(time.monotonic() - t0) * 1000.0,
            source="llm",
            **data,
        )

    def request_plan(self, object_label: str, size_cm: Optional[float] = None,
                     depth_m: Optional[float] = None, confidence: float = 1.0) -> bool:
        """Plan asynchronously; ``on_plan`` fires when done.

        Returns False (and does nothing) if a request is in flight or the
        cooldown hasn't elapsed, so the caller can recover its state.
        """
        with self._lock:
            now = time.monotonic()
            if self._pending or now - self._last_request_t < self.cooldown_s:
                return False
            self._pending = True
            self._last_request_t = now

        def worker():
            try:
                result = self.plan(object_label, size_cm, depth_m, confidence)
            finally:
                with self._lock:
                    self._pending = False
            if self.on_plan:
                self.on_plan(result)

        threading.Thread(target=worker, daemon=True).start()
        return True
