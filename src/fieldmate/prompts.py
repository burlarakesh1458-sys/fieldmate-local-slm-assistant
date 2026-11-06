"""Prompt templates. Kept in one module so prompt changes show up clearly in git diffs."""

from __future__ import annotations

OPS_RULES = """\
OPS MANUAL RULES
- Pre-flight decision is "no_go" if ANY of propellers/motors/gps/compass has an "issue",
  OR battery is below 30%, OR weather is not ok. Otherwise "go".
- A check the pilot did not mention or explicitly skipped is "not_checked" (not an issue).
- Defect severity:
  critical = fire/smoke/burning smell, battery swelling or puncture, structural crack in an
             arm or frame, motor/ESC/flight-controller failure in flight, loss of control.
  high     = any propeller damage, abnormal motor noise or vibration, compass or GPS errors,
             radio link loss / failsafe.
  medium   = camera/gimbal/payload problems, loose hardware, minor battery cell imbalance.
  low      = cosmetic only (scratches, stickers, dirty lens).
- grounded = true for high and critical, false for medium and low.
"""

SYSTEM_EXTRACT = (
    "You are FieldMate, an offline assistant for a drone operations team. "
    "Convert the pilot's messy note into JSON that matches the schema exactly. "
    "Use only facts in the note; use null when a value is not stated. "
    "Copy the aircraft ID from the note, upper-case (null if the note has none).\n\n" + OPS_RULES
)

SYSTEM_CHAT = (
    "You are FieldMate, an offline assistant for a drone operations team. "
    "Answer concisely and factually. If a fleet log is provided, answer ONLY from it and say "
    "'not in the log' when the answer is not there.\n\n" + OPS_RULES
)

FIELD_SPECS = {
    "preflight": (
        '{"aircraft_id": string|null, "battery_pct": integer 0-100|null, '
        '"propellers": "ok"|"issue"|"not_checked", "motors": "ok"|"issue"|"not_checked", '
        '"gps": "ok"|"issue"|"not_checked", "compass": "ok"|"issue"|"not_checked", '
        '"weather_ok": true|false|null, "decision": "go"|"no_go", "issues": [string]}'
    ),
    "defect": (
        '{"aircraft_id": string|null, "component": "propeller"|"motor"|"battery"|"esc"|"gps"|'
        '"compass"|"camera"|"gimbal"|"frame"|"landing_gear"|"radio"|"flight_controller"|"other", '
        '"severity": "low"|"medium"|"high"|"critical", "grounded": true|false, '
        '"symptom": string, "action_required": string}'
    ),
}

TASK_HINT = {
    "preflight": "Extract a PreflightReport from this pre-flight note.",
    "defect": "Extract a DefectReport from this maintenance / defect note.",
}


def extraction_messages(kind: str, text: str) -> list[dict[str, str]]:
    """System prompt is fixed per task kind so Ollama can reuse its KV-cache prefix."""
    system = f"{SYSTEM_EXTRACT}\nOUTPUT FIELDS ({kind}):\n{FIELD_SPECS[kind]}"
    user = f"{TASK_HINT[kind]}\n\nNOTE:\n{text.strip()}\n\nReturn only the JSON object."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def repair_message(error: str) -> dict[str, str]:
    """Error-feedback reprompt used by the single retry."""
    return {
        "role": "user",
        "content": (
            "Your previous JSON failed validation:\n"
            f"{error}\n"
            "Fix ONLY the problems listed, keep every other field faithful to the note, "
            "and return the corrected JSON object only."
        ),
    }


def chat_messages(question: str, context: str | None = None):
    user = question.strip()
    if context:
        user = f"FLEET LOG:\n{context.strip()}\n\nQUESTION: {user}"
    return [
        {"role": "system", "content": SYSTEM_CHAT},
        {"role": "user", "content": user},
    ]
