# Hiver SDE Intern Take-Home Assignment

## AI Customer Support Agent & Evaluation System (`@AppleSupport`)

## 1. Executive Summary

This project implements an AI customer-support agent for `@AppleSupport` using the Twitter Customer Support dataset.

For each customer query, the system:

1. Classifies the support intent.
2. Retrieves historically similar Apple Support resolutions.
3. Drafts a response grounded in retrieved historical evidence.
4. Decides whether to auto-handle the query or escalate it to a human, using explicit safety and business rules.
5. Evaluates the system against two baselines using automated metrics and an LLM-as-Judge rubric.

The final benchmark uses **200 human-verified examples** and a retrieval index with **zero overlap** with those evaluation examples.

---

## 2. Dataset and Brand Selection

### Brand: `@AppleSupport`

The project uses the Twitter Customer Support dataset and selects `@AppleSupport` because it provides a large number of technical customer-support conversations with historical brand resolutions.

The processed Apple subset contains:

- 204,772 `@AppleSupport` tweets
- 49,233 clean initial customer-query to brand-resolution pairs
- 49,033 historical resolution pairs in the retrieval index after removing evaluation examples

Only a subset of the full dataset is processed. Processing the complete dataset is not required for this assignment.

### Sampling and Golden Evaluation Set

A **200-example golden evaluation set** was created from the AppleSupport customer-query data.

Candidate labels were generated initially, followed by manual review of all 200 examples.

The final verified set is stored at:

data/human_verified_golden_set.csv


Each reviewed example contains:

- `human_final_intent`
- `human_final_escalation`
- `human_verified=True`

All 200 examples were manually verified.

The retrieval index excludes all 200 golden-set tweet IDs to prevent evaluation leakage.

---

## 3. Intent Taxonomy

The final taxonomy contains 8 categories:

| Intent | Description |
|---|---|
| `device_hardware_battery` | Physical hardware, battery, charging, screen, camera, microphone, overheating and related device issues |
| `software_update_bugs` | iOS/macOS/watchOS updates, OS bugs, crashes, freezes, restart loops and system issues |
| `account_appleid_security` | Apple ID lockouts, passwords, 2FA, account recovery and security issues |
| `billing_subscriptions_refunds` | Charges, refunds, subscriptions, payment issues and gift-card/payment problems |
| `connectivity_network_bluetooth` | Wi-Fi, Bluetooth, cellular connectivity, AirDrop and network issues |
| `app_thirdparty_behavior` | Third-party apps, App Store download/install issues, keyboard and emoji behavior |
| `audio_accessory_hardware` | AirPods, headphones, audio controls, Apple Watch accessories, dongles and adapters |
| `general_inquiry_other` | Generic inquiries, store/support hours, feature requests and cases without specific technical context |

Some customer queries contain signals for multiple categories. Deterministic priority rules are used to resolve these overlaps consistently.

---

## 4. System Architecture

The system follows a retrieval-augmented pipeline combining intent classification, historical-case retrieval, grounded response generation, and rule-based escalation.


                         Customer Tweet
                              |
                              v
                   +----------------------+
                   | Preprocessing         |
                   | Text cleanup          |
                   +----------+-----------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
     +-------------------+          +----------------------+
     | Intent            |          | Historical Retrieval |
     | Classification    |          | TF-IDF / Cosine      |
     | Rules + Priority  |          | Top-K similar cases  |
     +---------+---------+          +----------+-----------+
              |                               |
              +---------------+---------------+
                              |
                              v
                  +------------------------+
                  | Historical Resolutions |
                  | Used as Evidence       |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  | Grounded Response      |
                  | Generation              |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  | Escalation Engine      |
                  | Rules + Safety Checks  |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  | Final Response         |
                  | Auto-handle / Escalate|
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  | Evaluation Harness     |
                  | Metrics + LLM Judge    |
                  +------------------------+


### Pipeline Components

1. **Preprocessing**

   Cleans and normalizes the incoming customer tweet.

2. **Intent Classification**

   Assigns one of the eight support intents using deterministic rules and priority handling.

3. **Historical Retrieval**

   Retrieves similar resolved Apple Support cases using TF-IDF and cosine similarity.

4. **Grounded Response Generation**

   Uses retrieved historical cases as evidence when drafting the support response.

5. **Escalation Engine**

   Applies explicit safety and business rules.

   Examples include escalation for:

   - Financial or monetary disputes
   - Account security and lockout cases
   - Physical hardware damage requiring inspection
   - Cases requiring private account information or DM-based lookup

6. **Evaluation Harness**

   Evaluates intent classification, escalation decisions and response quality against the 200-example human-verified golden set.

---

## 5. Baselines

### Baseline 1 - Trivial Boilerplate

A deliberately simple baseline that uses generic customer-support language without meaningful historical retrieval.

This provides a reference point for how generic support responses perform.

### Baseline 2 - Simple TF-IDF + BM25

A lightweight retrieval/classification baseline using lexical similarity and historical-response retrieval.

### Main Agent - Grounded RAG Agent

The main system combines:

- Intent classification
- Historical retrieval
- Top-K evidence grounding
- Contextual response drafting
- Deterministic escalation rules

---

## 6. Final Benchmark Results

The final benchmark was run over the 200 human-verified golden examples.

| Agent | Intent Accuracy | Intent Macro F1 | Escalation F1 | LLM Judge Overall | Groundedness | Correctness | Tone | Safety | ROUGE-1 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline 1 - Trivial Boilerplate | 0.225 | 0.0459 | 0.4064 | 3.65 | 2.00 | 2.51 | 5.00 | 5.00 | 0.2360 | 0.1629 |
| Baseline 2 - TF-IDF + BM25 | 0.475 | 0.4908 | 0.4267 | 4.76 | 4.68 | 4.72 | 4.58 | 5.00 | 0.2844 | 0.2292 |
| Main Agent - Grounded RAG | 0.450 | 0.4715 | 0.3699 | 4.80 | 4.64 | 4.75 | 4.97 | 4.86 | 0.2778 | 0.2098 |

### Interpretation

The results show different strengths across the systems rather than a single metric dominating the evaluation.

The Main Agent receives a **4.80/5 LLM-as-Judge overall score** with strong groundedness, correctness, tone and safety scores.

The TF-IDF + BM25 baseline has higher intent accuracy and intent macro-F1 on this particular 200-example evaluation set.

The Main Agent's escalation F1 is lower than both baselines.

Therefore, the results are presented as **metric trade-offs rather than an overall ranking**.

---

## 7. LLM-as-Judge

The primary response-quality evaluation uses a rubric with four dimensions, each scored from 1 to 5.

### Evaluation Criteria

1. **Groundedness**

   Whether substantive claims in the generated response are supported by retrieved historical evidence.

2. **Correctness**

   Whether the response addresses the customer's actual issue.

3. **Tone**

   Whether the response is professional, concise and appropriate for customer support.

4. **Safety**

   Whether the response avoids unsupported guarantees, unsafe instructions and inappropriate requests for credentials or sensitive information.

The judge receives the customer query, generated response and retrieved historical evidence.

The evaluation prompt does not expose model identity, baseline identity, predicted intent or gold intent.

Generic boilerplate without supporting evidence is explicitly prevented from receiving a high groundedness or correctness score merely because it contains support-oriented language.

### Human vs. LLM-as-Judge Agreement

Human review in this project covers **intent and escalation labels**, not 1-5 response-quality ratings.

Therefore, a human-vs-LLM response-quality agreement statistic was **not computed**.

No fabricated or hard-coded human ratings are used.

This is an explicit limitation of the current evaluation and is preferable to reporting an unsupported agreement statistic.

---

## 8. Supporting ROUGE Metrics

ROUGE-1 and ROUGE-L are included only as supporting lexical metrics.

Customer-support responses can express the same resolution using substantially different wording. Therefore, word-overlap metrics are not treated as a direct measure of response quality.

The LLM-as-Judge rubric is used as the primary response-quality evaluation.

---

## 9. What Is Misleading About My Headline Number?

A single headline number can hide important trade-offs.

### 1. The 4.80/5 LLM-Judge score is not the same as customer satisfaction

The Main Agent's 4.80/5 score reflects the automated rubric used in this experiment.

It is not a measured production customer-satisfaction score or a substitute for human customer evaluation.

### 2. High response quality does not imply the highest intent accuracy

The Main Agent has an intent accuracy of **0.450**, while the TF-IDF + BM25 baseline has **0.475** on this evaluation set.

### 3. Escalation quality remains a weakness

The Main Agent has an escalation F1 of **0.3699**, compared with **0.4267** for the TF-IDF + BM25 baseline and **0.4064** for the trivial baseline.

### 4. ROUGE can be misleading

A semantically useful response can have low lexical overlap with the historical reference response.

### 5. The evaluation set is only 200 examples

The results are informative for this controlled benchmark but should not be treated as production-level performance estimates.

The dataset contains repetitive support language and the taxonomy is specific to this experiment.

---

## 10. Top 5 Failure Modes

### 1. Multi-intent queries

**Example:** A customer reports battery drain after an iOS update and also mentions Wi-Fi disconnections.

**Hypothesis:** The current single-label intent design forces multiple issues into one category.

**Potential improvement:** Multi-label intent detection followed by issue-specific retrieval.

### 2. Vague inquiries

**Example:** A customer asks whether Apple Support lines are closed.

**Hypothesis:** There may be insufficient technical context for retrieval or intent classification.

**Potential improvement:** Dedicated routing for informational and support-hours queries.

### 3. Account-recovery and private-account cases

**Example:** A customer has been waiting for an account-recovery code.

**Hypothesis:** These cases require escalation even when their wording resembles ordinary troubleshooting.

**Potential improvement:** Expand contextual security and escalation rules.

### 4. Noisy text, emojis and typos

**Example:** Tweets containing Unicode emoji, spelling errors or unusual formatting around technical terms.

**Hypothesis:** Lexical features can become less reliable when key words are noisy.

**Potential improvement:** Robust multilingual and semantic normalization with embedding-based classification.

### 5. Taxonomy overlap

**Example:** A battery problem that begins immediately after an operating-system update.

**Hypothesis:** Hardware, software and connectivity signals can legitimately co-occur.

**Potential improvement:** Multi-label classification or a learned routing model instead of a single deterministic priority.

---

## 11. Key Decision Log

1. Selected `@AppleSupport` because of the volume and technical troubleshooting content available in the dataset.
2. Used initial customer tweets as the starting point for support-resolution pairing.
3. Paired inbound customer queries with corresponding historical brand responses.
4. Created a separate candidate golden set before human verification.
5. Manually verified all 200 golden-set examples.
6. Used an 8-category taxonomy after empirical overlap analysis.
7. Excluded all 200 evaluation tweet IDs from the retrieval index.
8. Used a lightweight TF-IDF/BM25 retrieval approach to keep reproduction fast.
9. Added deterministic priority rules for overlapping intents.
10. Added explicit escalation rules for monetary disputes, account security, physical damage and private information.
11. Added disk caching to avoid repeated computation and API calls.
12. Used an LLM-as-Judge rubric rather than relying only on lexical overlap.
13. Added a strict rule preventing generic boilerplate from receiving high groundedness/correctness scores without evidence.
14. Kept ROUGE as a supporting metric rather than the primary response-quality measure.
15. Did not fabricate human-vs-LLM agreement statistics when the human review data did not contain matching 1-5 response-quality ratings.

---

## 12. Reproducibility

From the project root:

```bash
python -m pip install pandas scikit-learn sentence-transformers rouge-score kagglehub
python download_dataset.py
python explore_apple_data.py
python prepare_golden_candidate_set.py
python run_pipeline.py
```

The final evaluation pipeline is lightweight and the benchmark runtime is approximately 10-15 seconds after the required data and retrieval index are prepared.

### Final Evaluation Data

data/human_verified_golden_set.csv


### Final Benchmark Output

data/benchmark_results.csv

### Retrieval Index

data/apple_retrieval_index.csv


### Integrity Checks

The final integrity audit confirms:

- 200 verified examples
- No duplicate tweet IDs
- No `UNVERIFIED` labels
- Valid 8-intent taxonomy
- Valid escalation booleans
- Zero retrieval leakage
- Evaluator uses the human-verified golden set
- Model-blind evaluation prompt
- Both baselines and the main agent are present
- Reproduction is under 15 minutes

---

## 13. What I Would Do With One More Week

1. Add a semantic cross-encoder reranker to improve retrieval precision.
2. Introduce multi-label intent classification for multi-intent customer queries.
3. Add multilingual query handling.
4. Add structured output validation for generated responses.
5. Add multi-turn conversation memory.
6. Build a larger, independently reviewed response-quality evaluation subset with human 1-5 ratings so human-vs-LLM judge agreement can be measured directly.
7. Evaluate on temporally separated data to test robustness to changing product and software issues.
8. Integrate the agent with a production support workflow such as a ticketing or customer-support system.

---
## 14. Project Structure


hiver-sde-assignment/
|
|-- README.md
|-- .gitignore
|-- run_pipeline.py
|-- download_dataset.py
|-- explore_apple_data.py
|-- prepare_golden_candidate_set.py
|-- app_review.py
|-- audit_evaluation.py
|-- review_cli.py
|
|-- src/
|   |-- agent.py
|   |-- cache.py
|   |-- config.py
|   |-- evaluator.py
|   |-- intent_classifier.py
|   |-- llm_judge.py
|   |-- retriever.py
|
|-- scripts/
|   |-- extract_intent_guide.py
|   |-- final_integrity_check.py
|   |-- review_golden_set.py
|   |-- show_golden_set_preview.py
|   |-- test_judge_10_examples.py
|   |-- validate_workflow.py
|
|-- data/
|   |-- applesupport_tweets.csv
|   |-- apple_resolution_pairs.csv
|   |-- apple_retrieval_index.csv
|   |-- candidate_golden_set.csv
|   |-- human_review.csv
|   |-- human_verified_golden_set.csv
|   |-- benchmark_results.csv
