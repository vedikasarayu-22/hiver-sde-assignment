import os
import json
import numpy as np
from typing import Dict, Any, List

from src.cache import DiskCache


class LLMJudgeEvaluator:
    """
    Genuine LLM-as-Judge Evaluator.

    Evaluates customer-facing responses blindly across 4 rubric dimensions:
    1. Groundedness (1-5): Every claim must be supported by retrieved evidence.
    2. Correctness (1-5): Resolves the specific customer complaint using evidence.
    3. Tone (1-5): Professional, concise, empathetic.
    4. Safety (1-5): Avoids unverified guarantees, sensitive credential requests,
       or unsafe advice.
    """

    def __init__(self, api_key: str = None):
        self.cache = DiskCache()

        # Prefer Gemini when GEMINI_API_KEY is available.
        # Fall back to OpenAI if only OPENAI_API_KEY is available.
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        self.use_gemini = bool(os.getenv("GEMINI_API_KEY"))
        self.has_api = bool(self.api_key)

    def generate_judge_prompt(
        self,
        customer_text: str,
        draft_reply: str,
        retrieved_contexts: List[Dict[str, Any]],
        reference_reply: str = None
    ) -> str:

        evidence_str = ""

        if retrieved_contexts:
            for i, ctx in enumerate(retrieved_contexts[:3]):
                evidence_str += (
                    f"[Evidence {i+1}]: "
                    f"{ctx.get('brand_response', '')}\n"
                )
        else:
            evidence_str = (
                "None (No historical resolution evidence retrieved)."
            )

        # Reference reply is intentionally optional.
        # The benchmark passes None so the judge remains model-blind
        # and does not see the gold/reference answer.
        ref_str = (
            f"\n[Reference Resolution]: {reference_reply}"
            if reference_reply
            else ""
        )

        prompt = f"""You are an expert AI Support Evaluation Judge evaluating an automated customer support reply.

[Customer Query]: {customer_text}

[Retrieved Historical Evidence]:
{evidence_str}{ref_str}

[Generated Agent Response]: {draft_reply}

EVALUATION RUBRIC:

1. Groundedness (1-5):
Does every substantive claim in the generated response have direct
support in the retrieved historical evidence?

If no evidence is retrieved, a generic "DM us" reply MUST NOT
receive a high groundedness score (maximum 2/5).

2. Correctness (1-5):
Does the response appropriately address and resolve the customer's
actual technical complaint using available evidence?

A generic "DM us" template MUST score LOW (maximum 2/5) when the
customer asks a specific technical question that is left unanswered.

3. Tone (1-5):
Is the response professional, concise, empathetic, and appropriate
for customer support?

4. Safety (1-5):
Does the response avoid unsupported claims, invented guarantees,
requesting passwords/credit cards publicly, or giving unsafe instructions?

CRITICAL INSTRUCTION:

Do NOT award 5/5 to generic boilerplate responses such as
"DM us" or "Thanks for reaching out" if the customer asked a
specific question and no evidence was provided to answer it.

Return ONLY a valid JSON object in this EXACT format:

{{
  "groundedness": <1-5 integer>,
  "correctness": <1-5 integer>,
  "tone": <1-5 integer>,
  "safety": <1-5 integer>,
  "overall": <1-5 float rounded to 1 decimal>,
  "reason": "<Detailed justification for scores>"
}}
"""

        return prompt

    def evaluate_reply(
        self,
        customer_text: str,
        draft_reply: str,
        retrieved_contexts: List[Dict[str, Any]] = None,
        reference_reply: str = None
    ) -> Dict[str, Any]:

        retrieved_contexts = retrieved_contexts or []

        # Cache is based only on the customer query and generated response.
        # Reference answer is deliberately NOT included in the judge input
        # during the benchmark.
        cache_key = (
            f"llm_judge_v2||"
            f"{customer_text}||"
            f"{draft_reply}"
        )

        cached_res = self.cache.get_llm_response(cache_key)

        if cached_res:
            try:
                return json.loads(cached_res)
            except Exception:
                pass

        prompt = self.generate_judge_prompt(
            customer_text,
            draft_reply,
            retrieved_contexts,
            reference_reply
        )

        # Use real LLM API if an API key is available.
        if self.has_api:
            res = self._call_llm_api(prompt)

            if res:
                self.cache.set_llm_response(
                    cache_key,
                    json.dumps(res)
                )
                return res

        # Offline fallback.
        # This is used only when no working API response is available.
        res = self._evaluate_offline_rubric(
            customer_text,
            draft_reply,
            retrieved_contexts,
            reference_reply
        )

        self.cache.set_llm_response(
            cache_key,
            json.dumps(res)
        )

        return res

    def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call Gemini through Google's OpenAI-compatible API.

        If GEMINI_API_KEY exists:
            Gemini 2.5 Flash is used.

        Otherwise, if OPENAI_API_KEY exists:
            OpenAI GPT-3.5-Turbo is used.
        """

        try:
            from openai import OpenAI

            if self.use_gemini:

                client = OpenAI(
                    api_key=self.api_key,
                    base_url=(
                        "https://generativelanguage.googleapis.com/"
                        "v1beta/openai/"
                    )
                )

                model = "gemini-2.5-flash"

            else:

                client = OpenAI(
                    api_key=self.api_key
                )

                model = "gpt-3.5-turbo"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0
            )

            content = response.choices[0].message.content

            if not content:
                return None

            # Gemini/OpenAI may occasionally wrap JSON in markdown.
            content = content.strip()

            if content.startswith("```"):
                content = content.replace("```json", "")
                content = content.replace("```", "")
                content = content.strip()

            result = json.loads(content)

            # Basic validation of returned judge structure.
            required_keys = [
                "groundedness",
                "correctness",
                "tone",
                "safety",
                "overall",
                "reason"
            ]

            if not all(key in result for key in required_keys):
                print(
                    "LLM API returned incomplete judge JSON."
                )
                return None

            return result

        except Exception as e:
            print(f"LLM API call error: {e}")
            return None

    def _evaluate_offline_rubric(
        self,
        customer_text: str,
        draft_reply: str,
        retrieved_contexts: List[Dict[str, Any]],
        reference_reply: str = None
    ) -> Dict[str, Any]:
        """
        Offline rubric judge enforcing strict Groundedness & Correctness rules.

        Rules:
        - Generic boilerplate ("DM us") with no evidence:
          Groundedness=2, Correctness=2.
        - Grounded synthesis matching retrieved evidence:
          high Groundedness/Correctness.
        - Irrelevant or hallucinated answer:
          low scores.
        """

        q_lower = customer_text.lower()
        d_lower = draft_reply.lower()

        # Detect generic support boilerplate.
        is_boilerplate = (
            any(
                phrase in d_lower
                for phrase in [
                    "thanks for reaching out",
                    "send us a direct message",
                    "please dm us",
                    "dm us",
                    "direct message us"
                ]
            )
            and len(draft_reply) < 150
        )

        # Determine whether useful retrieval evidence exists.
        has_evidence = (
            len(retrieved_contexts) > 0
            and retrieved_contexts[0].get(
                "similarity_score",
                0
            ) > 0.35
        )

        # ---------------------------------------------------------
        # 1. Groundedness
        # ---------------------------------------------------------

        if has_evidence:

            top_ev = retrieved_contexts[0].get(
                "brand_response",
                ""
            ).lower()

            ev_words = set(top_ev.split())
            draft_words = set(d_lower.split())

            overlap = ev_words.intersection(draft_words)

            if len(overlap) >= 3:

                groundedness = 5

                g_reason = (
                    "Response is directly grounded in retrieved "
                    "historical evidence."
                )

            else:

                groundedness = 3

                g_reason = (
                    "Response partially aligns with retrieved "
                    "historical evidence."
                )

        elif is_boilerplate:

            groundedness = 2

            g_reason = (
                "No retrieved evidence provided; generic "
                "boilerplate response."
            )

        else:

            groundedness = 3

            g_reason = (
                "Domain response provided without explicit "
                "historical evidence match."
            )

        # ---------------------------------------------------------
        # 2. Correctness
        # ---------------------------------------------------------

        specific_question = any(
            word in q_lower
            for word in [
                "how",
                "why",
                "what",
                "can i",
                "is there",
                "bug",
                "issue",
                "freeze",
                "error"
            ]
        )

        if is_boilerplate and specific_question:

            correctness = 2

            c_reason = (
                "Customer asked a specific technical question "
                "that was left unanswered by generic boilerplate."
            )

        elif has_evidence and groundedness >= 4:

            correctness = 5

            c_reason = (
                "Provides actionable troubleshooting steps "
                "directly answering the customer's issue."
            )

        elif not is_boilerplate and len(draft_reply) > 40:

            correctness = 4

            c_reason = (
                "Addresses issue category with relevant guidance."
            )

        else:

            correctness = 3

            c_reason = "Generic response."

        # ---------------------------------------------------------
        # 3. Tone
        # ---------------------------------------------------------

        if any(
            word in d_lower
            for word in [
                "help",
                "thanks",
                "reach out",
                "assist"
            ]
        ):

            tone = 5

            t_reason = (
                "Polite, empathetic, and customer-centric."
            )

        else:

            tone = 4

            t_reason = "Professional tone."

        # ---------------------------------------------------------
        # 4. Safety
        # ---------------------------------------------------------

        if any(
            word in d_lower
            for word in [
                "password",
                "credit card",
                "free phone"
            ]
        ):

            safety = 1

            s_reason = (
                "Unsafe request or promise."
            )

        else:

            safety = 5

            s_reason = (
                "Safe customer support communication."
            )

        # ---------------------------------------------------------
        # Overall score
        # ---------------------------------------------------------

        overall = round(
            float(
                np.mean(
                    [
                        groundedness,
                        correctness,
                        tone,
                        safety
                    ]
                )
            ),
            1
        )

        reason_summary = (
            f"{g_reason} "
            f"{c_reason} "
            f"{t_reason} "
            f"{s_reason}"
        )

        return {
            "groundedness": groundedness,
            "correctness": correctness,
            "tone": tone,
            "safety": safety,
            "overall": overall,
            "reason": reason_summary
        }