# rag/moteur/prompting.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PromptPack:
    variant: str
    version: str
    system: str
    user: str

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def as_text(self) -> str:
        return (
            f"--- PROMPT ({self.variant} / {self.version}) ---\n"
            f"[SYSTEM]\n{self.system}\n\n"
            f"[USER]\n{self.user}\n"
            f"--- END PROMPT ---"
        )


PROMPT_V1_SYSTEM = (
    "Tu es un assistant qui recommande des événements culturels en Nouvelle-Aquitaine.\n"
    "Tu dois répondre UNIQUEMENT avec les informations du CONTEXTE fourni.\n"
    "Si le contexte ne permet pas de répondre, dis-le clairement et propose une reformulation.\n"
    "Réponds en français, de manière concise et utile.\n"
    "Donne au maximum 5 recommandations.\n"
    "Pour chaque recommandation : titre, date(s), ville, et URL.\n"
    "Termine par une section 'Sources' listant les IDs [S1..] utilisés."
)

PROMPT_V2_SYSTEM = (
    "Tu es un assistant culturel. "
    "Règles: "
    "1) Réponds uniquement avec les informations du CONTEXTE (événements/sources). "
    "2) Tu peux utiliser l'HISTORIQUE uniquement pour comprendre les questions de suivi "
    "(ex: 'et demain', 'dans la même ville', préférences), mais jamais comme source factuelle d'événements. "
    "3) Respecte la DATE_DE_REFERENCE pour interpréter les expressions relatives (ex: 'ce week-end', 'demain'). "
    "4) Applique les CONTRAINTES_TEMPS si elles sont fournies. "
    "5) Propose au maximum 5 recommandations. "
    "6) Pour chaque recommandation, donne le titre, les date(s), la ville et l'URL. "
    "7) Termine par 'Sources:' avec les identifiants [S1], [S2]..."
)


def is_followup_question(question: str) -> bool:
    q = (question or "").lower()
    markers = [
        "et demain",
        "et ce soir",
        "et le lendemain",
        "dans la même ville",
        "dans la meme ville",
        "pareil",
        "comme avant",
        "autre chose",
        "d'autres",
        "d’autres",
        "encore",
        "plutôt",
        "plutot",
        "finalement",
        "au lieu de",
        "pas ça",
        "pas ca",
    ]
    return any(m in q for m in markers)


def build_prompt(
    *,
    variant: str,
    question: str,
    context_text: str,
    now_ref: datetime | None = None,
    intent: str | None = None,
    time_guidance: str | None = None,
    history_text: str | None = None,
) -> PromptPack:
    """
    Centralise la logique de prompt.
    - variant="v1" : prompt simple (proche de rag_chain.answer_question)
    - variant="v2" : prompt time-aware + historique (celui de ton engine actuel)
    """
    v = (variant or "v2").lower().strip()
    version = os.environ.get("P9_PROMPT_VERSION", "v2_1")

    if v == "v1":
        system = PROMPT_V1_SYSTEM
        user = f"QUESTION:\n{question}\n\nCONTEXTE:\n{context_text}\n\nRÉPONSE:"
        return PromptPack(variant="v1", version=version, system=system, user=user)

    # default: v2
    system = PROMPT_V2_SYSTEM

    user_parts: list[str] = []
    if now_ref is not None:
        user_parts.append(f"DATE_DE_REFERENCE (Europe/Paris): {now_ref.isoformat()}")
    if intent:
        user_parts.append(f"INTENTION_TEMPORELLE: {intent}")
    if time_guidance:
        user_parts.append(f"CONTRAINTES_TEMPS: {time_guidance}")
    if history_text and is_followup_question(question):
        user_parts.append(f"HISTORIQUE (dern. échanges):\n{history_text}")

    user_parts.append(f"QUESTION:\n{question}")
    user_parts.append(f"CONTEXTE:\n{context_text}")
    user_parts.append("RÉPONSE:")

    user = "\n\n".join(user_parts)

    return PromptPack(variant="v2", version=version, system=system, user=user)
