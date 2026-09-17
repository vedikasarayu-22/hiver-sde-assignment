import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from rouge_score import rouge_scorer

from src.config import GOLDEN_SET_PATH
from src.agent import TrivialBaselineAgent, SimpleBaselineAgent, GroundedAIAgent
from src.llm_judge import LLMJudgeEvaluator

class EvaluationHarness:
    def __init__(self, golden_set_path=GOLDEN_SET_PATH):
        if not os.path.exists(golden_set_path):
            raise FileNotFoundError(f"Golden evaluation set not found at: {golden_set_path}")
        self.df = pd.read_csv(golden_set_path)
        self.judge = LLMJudgeEvaluator()
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def evaluate_agent(self, agent, agent_name: str) -> Dict[str, Any]:
        y_true_intent = self.df['human_final_intent'].fillna('general_inquiry_other').tolist()
        y_true_escalate = self.df['human_final_escalation'].astype(bool).tolist()
        reference_replies = self.df['brand_response'].fillna('').tolist()

        y_pred_intent = []
        y_pred_escalate = []
        draft_replies = []
        
        judge_scores = []
        rouge1_scores = []
        rougeL_scores = []

        print(f"\nEvaluating {agent_name} over {len(self.df)} Golden Set examples...")

        for idx, row in self.df.iterrows():
            customer_text = row['customer_text']
            ref_reply = row['brand_response']
            
            output = agent.process(customer_text)
            
            y_pred_intent.append(output['predicted_intent'])
            y_pred_escalate.append(bool(output['escalate']))
            draft = output['draft_reply']
            draft_replies.append(draft)

            # LLM-as-Judge Primary Evaluation
            j_eval = self.judge.evaluate_reply(
    customer_text,
    draft,
    output.get('retrieved_contexts', []),
    ref_reply
)
            judge_scores.append(j_eval)

            # Supporting ROUGE metrics
            r_score = self.rouge.score(ref_reply, draft)
            rouge1_scores.append(r_score['rouge1'].fmeasure)
            rougeL_scores.append(r_score['rougeL'].fmeasure)

        # 1. Intent Metrics
        intent_acc = accuracy_score(y_true_intent, y_pred_intent)
        intent_f1 = f1_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        
        # 2. Escalation Metrics
        esc_acc = accuracy_score(y_true_escalate, y_pred_escalate)
        esc_prec = precision_score(y_true_escalate, y_pred_escalate, zero_division=0)
        esc_rec = recall_score(y_true_escalate, y_pred_escalate, zero_division=0)
        esc_f1 = f1_score(y_true_escalate, y_pred_escalate, zero_division=0)

        # 3. LLM-as-Judge Rubric Metrics (Primary Response Quality)
        mean_groundedness = round(float(np.mean([s['groundedness'] for s in judge_scores])), 2)
        mean_correctness = round(float(np.mean([s['correctness'] for s in judge_scores])), 2)
        mean_tone = round(float(np.mean([s['tone'] for s in judge_scores])), 2)
        mean_safety = round(float(np.mean([s['safety'] for s in judge_scores])), 2)
        mean_overall_judge = round(float(np.mean([s['overall'] for s in judge_scores])), 2)

        # 4. Supporting Lexical Metrics
        mean_rouge1 = round(float(np.mean(rouge1_scores)), 4)
        mean_rougeL = round(float(np.mean(rougeL_scores)), 4)

        return {
            'agent_name': agent_name,
            'intent_accuracy': round(intent_acc, 4),
            'intent_macro_f1': round(intent_f1, 4),
            'escalation_accuracy': round(esc_acc, 4),
            'escalation_precision': round(esc_prec, 4),
            'escalation_recall': round(esc_rec, 4),
            'escalation_f1': round(esc_f1, 4),
            'judge_overall_score': mean_overall_judge,
            'judge_groundedness': mean_groundedness,
            'judge_correctness': mean_correctness,
            'judge_tone': mean_tone,
            'judge_safety': mean_safety,
            'supporting_rouge1': mean_rouge1,
            'supporting_rougeL': mean_rougeL,
            'raw_judge_scores': [s['overall'] for s in judge_scores]
        }

    def run_full_benchmark(self) -> pd.DataFrame:
        agents = [
            (TrivialBaselineAgent(), "Baseline 1 (Trivial Boilerplate)"),
            (SimpleBaselineAgent(), "Baseline 2 (Simple TF-IDF + BM25)"),
            (GroundedAIAgent(), "Main Agent (Grounded RAG Agent)")
        ]

        results = []
        for agent_inst, name in agents:
            res = self.evaluate_agent(agent_inst, name)
            results.append(res)

        # Human agreement evaluation on 30-example subset
        # Not computed here because the human review set contains
        # intent and escalation labels, not human 1-5 response-quality ratings.
        # Do not use fabricated or hard-coded human ratings.
        print("\n" + "=" * 70)
        print("HUMAN VS. LLM-AS-JUDGE AGREEMENT")
        print("=" * 70)
        print("  Not computed: human review contains intent/escalation labels,")
        print("  not human response-quality ratings on the 1-5 judge rubric.")

        # Format clean summary dataframe
        summary_rows = []
        for r in results:
            summary_rows.append({
                'Agent': r['agent_name'],
                'Intent Acc': r['intent_accuracy'],
                'Intent Macro F1': r['intent_macro_f1'],
                'Escalation F1': r['escalation_f1'],
                'LLM Judge Overall (1-5)': r['judge_overall_score'],
                'Groundedness': r['judge_groundedness'],
                'Correctness': r['judge_correctness'],
                'Tone': r['judge_tone'],
                'Safety': r['judge_safety'],
                'ROUGE-1 (Supp)': r['supporting_rouge1'],
                'ROUGE-L (Supp)': r['supporting_rougeL']
            })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df
