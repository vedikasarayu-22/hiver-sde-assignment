import re
from typing import Dict, Any, List
from src.intent_classifier import MajorityBaselineClassifier, SimpleTfidfClassifier, RuleLLMClassifier
from src.retriever import HistoricalResolutionRetriever
from src.cache import DiskCache

class TrivialBaselineAgent:
    def __init__(self):
        self.classifier = MajorityBaselineClassifier()

    def process(self, customer_text: str) -> Dict[str, Any]:
        intent = self.classifier.predict(customer_text)
        reply = "Thanks for reaching out to Apple Support! Please send us a Direct Message with your device model and iOS version so we can assist you."
        return {
            'predicted_intent': intent,
            'draft_reply': reply,
            'escalate': True,
            'escalation_reason': "Default static fallback policy",
            'retrieved_contexts': []
        }


class SimpleBaselineAgent:
    def __init__(self):
        self.classifier = SimpleTfidfClassifier()
        self.retriever = HistoricalResolutionRetriever()

    def process(self, customer_text: str) -> Dict[str, Any]:
        intent = self.classifier.predict(customer_text)
        retrieved = self.retriever.retrieve(customer_text, top_k=1)
        
        if retrieved:
            draft_reply = retrieved[0]['brand_response']
        else:
            draft_reply = "Please DM us your device details so we can investigate."

        # Simple keyword escalation rule
        t_lower = str(customer_text).lower()
        reasons = []
        if any(w in t_lower for w in ['refund', 'money', 'card', 'billing', 'charged']):
            reasons.append("Billing/refund keyword detected")
        if any(w in t_lower for w in ['locked', 'hacked', 'security', 'stolen', 'apple id']):
            reasons.append("Account security keyword detected")
        if any(w in t_lower for w in ['crack', 'damaged', 'broken', 'repair']):
            reasons.append("Hardware repair keyword detected")

        escalate = len(reasons) > 0
        reason = "; ".join(reasons) if escalate else "Standard technical query auto-handled"

        return {
            'predicted_intent': intent,
            'draft_reply': draft_reply,
            'escalate': escalate,
            'escalation_reason': reason,
            'retrieved_contexts': retrieved
        }


class GroundedAIAgent:
    def __init__(self):
        self.classifier = RuleLLMClassifier()
        self.retriever = HistoricalResolutionRetriever()
        self.cache = DiskCache()

    def process(self, customer_text: str) -> Dict[str, Any]:
        intent = self.classifier.predict(customer_text)
        retrieved_contexts = self.retriever.retrieve(customer_text, top_k=3)

        # Grounded response synthesis
        t_lower = customer_text.lower()
        
        # Determine escalation & grounded response strategy
        escalate, reason = self._evaluate_escalation(customer_text, retrieved_contexts)
        
        draft_reply = self._generate_grounded_reply(customer_text, intent, retrieved_contexts, escalate)
        
        return {
            'predicted_intent': intent,
            'draft_reply': draft_reply,
            'escalate': escalate,
            'escalation_reason': reason,
            'retrieved_contexts': retrieved_contexts
        }

    def _evaluate_escalation(self, customer_text: str, contexts: List[Dict[str, Any]]):
        t_lower = customer_text.lower()
        reasons = []

        # 1. Financial/Monetary Dispute
        if any(w in t_lower for w in ['refund', 'money', 'charged twice', 'billing dispute', 'unauthorized purchase']):
            reasons.append("Financial transaction / refund request requires agent account verification")
            
        # 2. Account Security & Recovery
        if any(w in t_lower for w in ['hacked', 'stolen', 'locked out', 'disabled apple id', '2fa code']):
            reasons.append("Account security compromise / identity verification required")

        # 3. Physical Hardware Damage
        if any(w in t_lower for w in ['shattered screen', 'water damage', 'swollen battery', 'broken speaker']):
            reasons.append("Physical hardware repair inspection required at Genius Bar")

        # 4. DM Escalation Indicator in Top Retrieval
        if contexts:
            top_reply = contexts[0]['brand_response'].lower()
            if ('dm' in top_reply or 'direct message' in top_reply) and len(reasons) == 0:
                reasons.append("Historical resolutions for this specific issue consistently require private DM details")

        if reasons:
            return True, "; ".join(reasons)
        else:
            return False, "Self-service technical troubleshooting instructions provided"

    def _generate_grounded_reply(self, customer_text: str, intent: str, contexts: List[Dict[str, Any]], escalate: bool) -> str:
        # Check cache first
        cache_prompt = f"{customer_text}||{intent}||{escalate}"
        cached_res = self.cache.get_llm_response(cache_prompt)
        if cached_res:
            return cached_res

        # Dynamic grounding based on historical resolutions
        if contexts and contexts[0]['similarity_score'] > 0.35:
            ref_reply = contexts[0]['brand_response']
            # Synthesize grounded reply adhering to Apple persona
            if escalate:
                reply = f"We hear you and want to help resolve this. {ref_reply} Send us a DM with your iOS version and device model so we can take a closer look."
            else:
                reply = f"We're here to help! {ref_reply}"
        else:
            # Domain-specific grounded advice based on intent
            if intent == "software_update_bugs":
                reply = "We're here to help. Try restarting your device, ensuring you have a backup, and checking Settings > General > Software Update. If the issue persists, send us a DM!"
            elif intent == "device_hardware_battery":
                reply = "Battery life is important! Check Settings > Battery to review app usage. If your battery capacity is degraded, send us a DM so we can guide you on service options."
            elif intent == "connectivity_network_bluetooth":
                reply = "Let's get you reconnected! Try toggling Airplane Mode, restarting your Wi-Fi router, or resetting Network Settings in Settings > General > Reset."
            elif intent == "account_appleid_security":
                reply = "Account security is a top priority. Please visit iforgot.apple.com to reset your password or verify your Apple ID credentials. Send us a DM if you need further help!"
            elif intent == "billing_subscriptions_refunds":
                reply = "We can assist with billing questions. You can view your purchase history or request a refund at reportaproblem.apple.com. DM us if you need help navigating your account!"
            else:
                reply = "Thanks for reaching out to Apple Support! Please send us a Direct Message with your device details and we'll be happy to assist."

        self.cache.set_llm_response(cache_prompt, reply)
        return reply
