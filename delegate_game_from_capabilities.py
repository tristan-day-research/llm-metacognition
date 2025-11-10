"""
DelegateGameFromCapabilities - Cleaned, centralized, and logged.
Adds optional decision-only mode for delegate vs answer, with alternating mappings.

Features:
- Loads completed results file (Phase 1 from capabilities)
- Constructs Phase 1 simulated history (configurable)
- Selects Phase 1 and Phase 2 question sets to hit target accuracies
- Runs Phase 2 (delegate game) with multiple choice or short answer questions
- Centralizes prompts and run parameters; records run-level config for reproducibility
- Optional "decision-only" mode: choose Answer vs Delegate, mapping alternates per trial
- Summary logging (no heavy statistical analysis)
"""

import random
import time
import copy
import json
import os
import re
from base_game_class import BaseGameClass
from load_and_format_datasets import load_and_format_dataset
import string
import glob

# Optional filtering by answer types (for SimpleQA/SimpleMC)
PHASE1_TYPES = None  # e.g., ["Number", "Other", "Place"]
PHASE2_TYPES = None  # e.g., ["Date", "Person"]

class DelegateGameFromCapabilities(BaseGameClass):
    def __init__(
        self,
        subject_id,
        subject_name,
        is_human_player=False,
        completed_results_file=None,
        dataset="GPQA",
        n_trials_phase1=100,
        n_trials_phase2=100,
        teammate_accuracy_phase1=0.7,
        teammate_accuracy_phase2=0.7,
        feedback_config=None,
        override_subject_accuracy=None,
        override_subject_accuracy_game=None,
        randomize_phase1_answers=False,
        use_phase1_summary=True,
        use_phase1_history=True,
        redact_phase1_answers=False,
        initial_setup_explanation="",
        seed=None,
        temperature=0.0,
        resume_from=None,
        include_question_num=False,
        include_total_questions=False,
        decision_only=False,
        alternate_decision_mapping=True
    ):
        super().__init__(subject_id, subject_name, is_human_player, "delegate_game_logs")

        # Seed RNG
        self.seed = seed
        if self.seed is not None:
            self._log(f"Using random seed: {self.seed}")
            random.seed(self.seed)

        # Core configuration
        self.temperature = temperature
        self.completed_results_file = completed_results_file
        self.dataset = dataset
        self.n_trials_phase1 = n_trials_phase1
        self.n_trials_phase2 = n_trials_phase2
        self.teammate_accuracy_phase1 = teammate_accuracy_phase1
        self.teammate_accuracy_phase2 = teammate_accuracy_phase2
        self.override_subject_accuracy = override_subject_accuracy
        self.override_subject_accuracy_game = override_subject_accuracy_game
        self.randomize_phase1_answers = randomize_phase1_answers
        self.use_phase1_summary = use_phase1_summary
        self.use_phase1_history = use_phase1_history
        self.redact_phase1_answers = redact_phase1_answers
        self.initial_setup_explanation_template = initial_setup_explanation
        self.include_question_num = include_question_num
        self.include_total_questions = include_total_questions
        self.decision_only = decision_only
        self.alternate_decision_mapping = alternate_decision_mapping

        # State
        self.results = []
        self.phase1_questions = []
        self.phase2_questions = []
        self.subject_accuracy_phase1 = None
        self.true_subject_accuracy = None
        self.phase2_score = None
        self.phase2_accuracy = None
        self.stored_message_history = []
        self.phase1_simulated_history = []
        self.is_short_answer = False
        self.teammate_phase1_answers = {}
        self.teammate_phase2_answers = {}
        self.all_correct_questions = []
        self.all_incorrect_questions = []
        self.pending_feedback = ""

        self.decision_only_options = ["A", "D"] if not self.alternate_decision_mapping else ["1", "2"]

        # Prompts (centralized)
        self.prompts = {
            # Phase headers
            "phase1_header": "="*10 + " Starting Phase 1: Answer the Questions " + "="*10 + "\n",
            "phase2_header_game": "="*10 + " Game: Answer or Delegate " + "="*10 + "\n",
            "phase2_header_phase2": "="*10 + " Phase 2: Answer or Delegate " + "="*10 + "\n",

            # Human input prompts
            "sa_answer_prompt": "Your answer: ",
            "sa_choice_with_delegate": "Your choice (answer succinctly or T=Teammate): ",
            "mc_answer_prompt": "Your answer (A, B, C, D): ",
            "mc_choice_with_delegate": "Respond ONLY with 'A', 'B', 'C', 'D', or 'T'\nYour choice (A, B, C, D, or T=Teammate): ",

            # LLM instruction lines shown in setup_text
            "mc_setup_prompt": "For each question, enter your answer (A, B, C, D) or 'T' to let your teammate answer. Do NOT output any other text.",
            "sa_setup_prompt": "For each question, answer as succinctly as possible or respond with 'T' to let your teammate answer.\n",
            "llm_sa_sysprompt": "\nRespond ONLY with your answer or 'T'.\n",
            "llm_mc_sysprompt": "\nOutput ONLY the letter of your choice: A, B, C, D, or T. Do NOT output any other text. Do not offer any explanation or justification.\n",

            # Decision-only mode
            "decision_only_sysprompt": f"Respond ONLY with '{self.decision_only_options[0]}' or '{self.decision_only_options[1]}'. Do NOT output any other text. Do not offer any explanation or justification.\n",
            #"decision_only_choice_line1": f"Choices:\n  {self.decision_only_options[0]}: {ONE_MEANING}\n  {self.decision_only_options[1]}: {TWO_MEANING}\nRespond ONLY with '{self.decision_only_options[0]}' or '{self.decision_only_options[1]}'.\nYour choice ('{self.decision_only_options[0]}' or '{self.decision_only_options[1]}'): ",
            "decision_only_choice_line": f"Choices:\n  {self.decision_only_options[0]}: {{ONE_MEANING}}\n  {self.decision_only_options[1]}: {{TWO_MEANING}}\nRespond ONLY with '{self.decision_only_options[0]}' or '{self.decision_only_options[1]}'.\nYour choice ('{self.decision_only_options[0]}' or '{self.decision_only_options[1]}'): ",

            # Counters/feedback
            "feedback_teammate_delegation": "--> Delegating to teammate...",
            "feedback_subject_answer_prefix": "--> Your answer: ",
        }

        # Feedback configuration defaults
        self.feedback_config = {
            'phase1_subject_feedback': True,
            'phase1_teammate_feedback': True,
            'phase2_subject_feedback': False,
            'phase2_teammate_feedback': False,
            'show_answer_with_correctness': True,
        }
        if feedback_config:
            self.feedback_config.update(feedback_config)

        # Load completed results, derive question pools, teammate answers
        self._load_completed_results()

        # Resolve setup explanation (use the provided template verbatim)
        self.initial_setup_explanation = self.initial_setup_explanation_template

        # Resume
        if resume_from:
            self._log(f"Resuming from: {resume_from}")
            try:
                with open(resume_from, 'r', encoding='utf-8') as f:
                    res = json.load(f)
                self.completed_results = res.get("results")
            except Exception as e:
                self._log(f"Error resuming from {resume_from}: {e}")
                raise ValueError(f"Could not resume from {resume_from}: {e}")
        else:
            self.completed_results = None

        # Determine static call args for LLM path
        max_tokens_used = None if ('opus-4' in self.subject_name or 'sonnet-4' in self.subject_name or '3-5-sonnet' in self.subject_name) else 1

        self.get_llm_answer_static_args = {
            "keep_appending": (False if not self.feedback_config['phase2_teammate_feedback'] and not self.feedback_config['phase2_subject_feedback'] else True),
            "message_history": [],
            "MAX_TOKENS": max_tokens_used,
            "temp": self.temperature,
            "accept_any": False if not self.is_short_answer and not self.decision_only else True
        }

        # Record run-level parameters for reproducibility
        self.run_parameters = {
            "dataset": self.dataset,
            "completed_results_file": self.completed_results_file,
            "n_trials_phase1": self.n_trials_phase1,
            "n_trials_phase2": self.n_trials_phase2,
            "teammate_accuracy_phase1": self.teammate_accuracy_phase1,
            "teammate_accuracy_phase2": self.teammate_accuracy_phase2,
            "override_subject_accuracy": self.override_subject_accuracy,
            "override_subject_accuracy_game": self.override_subject_accuracy_game,
            "randomize_phase1_answers": self.randomize_phase1_answers,
            "use_phase1_summary": self.use_phase1_summary,
            "use_phase1_history": self.use_phase1_history,
            "redact_phase1_answers": self.redact_phase1_answers,
            "include_question_num": self.include_question_num,
            "include_total_questions": self.include_total_questions,
            "decision_only": self.decision_only,
            "alternate_decision_mapping": self.alternate_decision_mapping,
            "is_human_player": self.is_human_player,
            "temperature": self.temperature,
            "seed": self.seed,
            "is_short_answer": self.is_short_answer,
            "prompts_used": self.prompts,
            "get_llm_answer_static_args": self.get_llm_answer_static_args
        }

    def _load_completed_results(self):
        """Load completed results and set up question pools and teammates."""
        if not self.completed_results_file or not os.path.exists(self.completed_results_file):
            raise ValueError(f"Completed results file not found: {self.completed_results_file}")

        try:
            self._log(f"Loading completed results from: {self.completed_results_file}")
            with open(self.completed_results_file, 'r', encoding='utf-8') as f:
                self.completed_data = json.load(f)

            if "results" not in self.completed_data or not isinstance(self.completed_data["results"], dict):
                raise ValueError("Invalid completed results file: missing or invalid 'results' field")
            if "accuracy" not in self.completed_data:
                raise ValueError("Invalid completed results file: missing 'accuracy' field")

            # True subject accuracy from completed results (Phase 1 actual)
            self.true_subject_accuracy = float(self.completed_data["accuracy"])
            self._log(f"True subject accuracy from completed results: {self.true_subject_accuracy:.2%}")

            # Determine Phase 1 subject accuracy to simulate (override or true)
            if self.override_subject_accuracy is not None:
                self.subject_accuracy_phase1 = float(self.override_subject_accuracy)
                self._log(f"Using override subject accuracy for Phase 1: {self.subject_accuracy_phase1:.2%}")
            else:
                self.subject_accuracy_phase1 = self.true_subject_accuracy
                self._log(f"Using true subject accuracy for Phase 1: {self.subject_accuracy_phase1:.2%}")

            # Determine if SA or MC
            self._determine_question_type()

            # Build correct/incorrect pools
            self._separate_questions_by_correctness()

            # Prepare Phase 1 and Phase 2 selections
            self._prepare_phase1_questions()
            self._prepare_phase2_questions()

            # Predetermine teammate answers for both phases
            self._predetermine_teammate_answers(phase=1)
            self._predetermine_teammate_answers(phase=2)

            self._log(f"Selected {len(self.phase1_questions)} questions for Phase 1")
            self._log(f"Selected {len(self.phase2_questions)} questions for Phase 2")
            self._log(f"Question type: {'Short Answer' if self.is_short_answer else 'Multiple Choice'}")

        except Exception as e:
            raise ValueError(f"Error loading completed results data: {e}")

    def _determine_question_type(self):
        """Determine if results represent MC or SA."""
        result = next(iter(self.completed_data["results"].values()))
        first = result['question'] if isinstance(result['question'], dict) else result
        self.is_short_answer = not ("options" in first and isinstance(first["options"], dict) and len(first["options"]) > 0)

    def _separate_questions_by_correctness(self):
        """Build pools of correct and incorrect questions from completed results."""
        self.all_correct_questions = []
        self.all_incorrect_questions = []

        if self.dataset == "GPQA":
            gpqa_all = load_and_format_datasetsafe("GPQA")
        else:
            gpqa_all = []

        for q_id, result_item in self.completed_data["results"].items():
            # Optional GPQA filtering by domain/difficulty via subject_id suffix
            if self.dataset == "GPQA":
                feature = next((x for x in gpqa_all if x.get('id') == q_id), None)
                domain = feature.get('high_level_domain') if feature else None
                difficulty = feature.get('difficulty_score') if feature else None
                if "_nobio" in self.subject_id and domain and str(domain).lower() == "biology":
                    continue
                if "_noeasy" in self.subject_id and difficulty and difficulty < 2:
                    continue

            is_corr = result_item.get("is_correct")
            if is_corr not in (True, False):
                continue

            result_q = result_item['question'] if isinstance(result_item['question'], dict) else result_item
            qdata = {
                "id": q_id,
                "is_correct": bool(is_corr),
                "subject_answer": result_item.get("subject_answer", ""),
                "probs": result_item.get("probs"),
                "question": result_q.get("question", "N/A"),
                "options": result_q.get("options", {}),
                "correct_answer": result_q.get("correct_answer_label", result_q.get("correct_answer", "N/A")),
            }

            if qdata["is_correct"]:
                self.all_correct_questions.append(qdata)
            else:
                self.all_incorrect_questions.append(qdata)

        random.shuffle(self.all_correct_questions)
        random.shuffle(self.all_incorrect_questions)
        self._log(f"Separated questions: {len(self.all_correct_questions)} correct, {len(self.all_incorrect_questions)} incorrect")

    def _prepare_phase1_questions(self):
        """Select Phase 1 questions to hit target subject accuracy."""
        need_correct = int(round(self.subject_accuracy_phase1 * self.n_trials_phase1))
        need_incorrect = self.n_trials_phase1 - need_correct
        self._log(f"Selecting {need_correct} correct and {need_incorrect} incorrect questions for Phase 1")

        # Optional filtering by answer types (for SimpleQA)
        if PHASE1_TYPES and self.dataset == "SimpleQA":
            sqa_all = load_and_format_dataset("SimpleQA")
            lookup = {x['id']: x.get('answer_type') for x in sqa_all}
            phase1_corr = [q for q in self.all_correct_questions if lookup.get(q["id"]) in PHASE1_TYPES][:need_correct]
            phase1_inc = [q for q in self.all_incorrect_questions if lookup.get(q["id"]) in PHASE1_TYPES][:need_incorrect]
        else:
            phase1_corr = self.all_correct_questions[:need_correct]
            phase1_inc = self.all_incorrect_questions[:need_incorrect]

        self.phase1_questions = phase1_corr + phase1_inc
        random.shuffle(self.phase1_questions)
        self.phase1_question_ids = set(q["id"] for q in self.phase1_questions)
        actual_correct = sum(1 for q in self.phase1_questions if q["is_correct"])
        self.subject_accuracy_phase1 = (actual_correct / len(self.phase1_questions)) if self.phase1_questions else 0.0
        self._log(f"Phase 1 selected {len(self.phase1_questions)} with subject accuracy {self.subject_accuracy_phase1:.2%}")

    def _prepare_phase2_questions(self):
        """Select Phase 2 questions to approximate true subject accuracy, non-overlapping with Phase 1."""
        # Optional filtering by answer types (for SimpleQA)
        if PHASE2_TYPES and self.dataset == "SimpleQA":
            sqa_all = load_and_format_dataset("SimpleQA")
            lookup = {x['id']: x.get('answer_type') for x in sqa_all}
            remaining_correct = [q for q in self.all_correct_questions if (q["id"] not in self.phase1_question_ids) and (lookup.get(q["id"]) in PHASE2_TYPES)]
            remaining_incorrect = [q for q in self.all_incorrect_questions if (q["id"] not in self.phase1_question_ids) and (lookup.get(q["id"]) in PHASE2_TYPES)]
        else:
            remaining_correct = [q for q in self.all_correct_questions if q["id"] not in self.phase1_question_ids]
            remaining_incorrect = [q for q in self.all_incorrect_questions if q["id"] not in self.phase1_question_ids]

        total_remaining = len(remaining_correct) + len(remaining_incorrect)
        if total_remaining == 0:
            self.phase2_questions = []
            self.n_trials_phase2 = 0
            self._log("No questions available for Phase 2.")
            return

        if self.n_trials_phase2 > total_remaining:
            self._log(f"Requested {self.n_trials_phase2} Phase 2 questions but only {total_remaining} remaining; using all.")
            self.n_trials_phase2 = total_remaining

        # Optionally override target accuracy for Phase 2
        phase2_target_acc = self.override_subject_accuracy_game if self.override_subject_accuracy_game is not None else self.true_subject_accuracy

        target_correct = int(round(phase2_target_acc * self.n_trials_phase2))
        target_incorrect = self.n_trials_phase2 - target_correct

        selected_correct = remaining_correct[:min(target_correct, len(remaining_correct))]
        selected_incorrect = remaining_incorrect[:min(target_incorrect, len(remaining_incorrect))]
        selected = selected_correct + selected_incorrect

        # Fill if short
        while len(selected) < self.n_trials_phase2 and len(selected_correct) < len(remaining_correct):
            selected_correct.append(remaining_correct[len(selected_correct)])
            selected.append(selected_correct[-1])
        while len(selected) < self.n_trials_phase2 and len(selected_incorrect) < len(remaining_incorrect):
            selected_incorrect.append(remaining_incorrect[len(selected_incorrect)])
            selected.append(selected_incorrect[-1])

        random.shuffle(selected)
        self.phase2_questions = selected
        self.n_trials_phase2 = len(self.phase2_questions)
        if self.n_trials_phase2 > 0:
            actual_corr = sum(1 for q in self.phase2_questions if q["is_correct"])
            actual_acc = actual_corr / self.n_trials_phase2
            self._log(f"Phase 2 selected {self.n_trials_phase2} questions; target acc {phase2_target_acc:.2%}, actual {actual_acc:.2%}")

    def _predetermine_teammate_answers(self, phase=1):
        """Plan teammate answers to hit target accuracy exactly for the selected set."""
        qs = self.phase1_questions if phase == 1 else self.phase2_questions
        target = self.teammate_accuracy_phase1 if phase == 1 else self.teammate_accuracy_phase2
        correct_count = int(round(target * len(qs)))
        indices = list(range(len(qs)))
        random.shuffle(indices)
        correct_indices = set(indices[:correct_count])

        answers = {}
        for i, q in enumerate(qs):
            if self.is_short_answer:
                if i in correct_indices:
                    answers[q["id"]] = (q["correct_answer"], True)
                else:
                    answers[q["id"]] = (f"Wrong answer for {q['id']}", False)
            else:
                options = list(q["options"].keys())
                if i in correct_indices:
                    answers[q["id"]] = (q["correct_answer"], True)
                else:
                    wrong = [o for o in options if o != q["correct_answer"]]
                    answers[q["id"]] = (random.choice(wrong), False)
        if phase == 1:
            self.teammate_phase1_answers = answers
        else:
            self.teammate_phase2_answers = answers

        self._log(f"Predetermined teammate answers for Phase {phase}: {len(answers)} (correct={correct_count})")

    def _present_question_with_indices(self, question, i, total):
        """present_question wrapper honoring include_question_num/total toggles."""
        if self.include_question_num and self.include_total_questions:
            return self._present_question(question, i, total)
        elif self.include_question_num:
            return self._present_question(question, i)
        else:
            return self._present_question(question)

    def _create_simulated_phase1_history(self):
        """Construct Phase 1 history (if enabled)."""
        history = []
        self.pending_feedback = ""
        startup = self.initial_setup_explanation + "\n\n" + self.prompts["phase1_header"]
        prompt = (self.prompts["mc_answer_prompt"] if not self.is_short_answer else self.prompts["sa_answer_prompt"])

        if self.use_phase1_history:
            if self.randomize_phase1_answers:
                n = self.n_trials_phase1
                ones = int(round(self.subject_accuracy_phase1 * n))
                zeros = n - ones
                dummy = [1]*ones + [0]*zeros
                random.shuffle(dummy)
            for idx, q in enumerate(self.phase1_questions, start=1):
                q_text = self._present_question_with_indices(q, idx, len(self.phase1_questions)) + "\n" + prompt
                if self.pending_feedback:
                    q_text = self.pending_feedback + "\n\n" + q_text
                    self.pending_feedback = ""

                user_msg = {"role": "user", "content": startup + q_text}
                startup = ""
                if self.redact_phase1_answers:
                    assistant_msg = {"role": "assistant", "content": "[redacted]"}
                else:
                    assistant_msg = {"role": "assistant", "content": q.get("subject_answer", "")}

                history.append(user_msg)
                history.append(assistant_msg)

                # Schedule feedback to prepend to next question (not separate messages)
                subj_corr = q.get("is_correct", False) if not self.randomize_phase1_answers else bool(dummy[idx-1])
                feedback_lines = []

                if self.feedback_config['phase1_subject_feedback']:
                    subj_ans = q.get("subject_answer", "")
                    fb = f"Your answer: {subj_ans}" if self.feedback_config['show_answer_with_correctness'] else "Your answer:"
                    fb += f" ({'Correct' if subj_corr else 'Incorrect'})"
                    feedback_lines.append(fb)

                if self.feedback_config['phase1_teammate_feedback']:
                    t_ans, t_corr = self.teammate_phase1_answers[q["id"]]
                    fb = f"Teammate's answer: {t_ans}" if self.feedback_config['show_answer_with_correctness'] else "Teammate's answer:"
                    fb += f" ({'Correct' if t_corr else 'Incorrect'})"
                    feedback_lines.append(fb)

                if feedback_lines:
                    self.pending_feedback = "\n".join(feedback_lines)

        self.phase1_simulated_history = history
        self._log(f"Created simulated Phase 1 history with {len(history)} messages")
        return history

    def _parse_subject_decision(self, resp, options):
        """Normalize response to one of the allowed tokens if possible."""
        if resp is None:
            return None
        if len(resp.rstrip(string.whitespace + string.punctuation)) == 0:
            return resp
        arr = resp.strip().upper().rstrip(string.whitespace + string.punctuation)
        if arr and arr[0] in options:
            return arr[0]
        if arr and arr[-1] in options:
            return arr[-1]
        return resp

    def _record_trial(self, phase, trial_num, q_data, **kwargs):
        """Append a trial result."""
        trial = {
            "subject_id": self.subject_id,
            "phase": phase,
            "trial_in_phase": trial_num,
            "question_id": q_data.get("id"),
            "question_text": q_data.get("question"),
            "correct_answer": q_data.get("correct_answer"),
            "timestamp": time.time(),
            "team_answer": None,
            "team_correct": None,
            "delegation_choice": None,
            "subject_answer": None,
            "subject_correct": None,
            "teammate_answer": None,
            "teammate_correct": None,
            "probs": None,
            "options": copy.deepcopy(q_data.get("options", {})) if not self.is_short_answer else None
        }
        trial.update(kwargs)
        self.results.append(trial)

    def run_delegate_game(self):
        """Run the game: build Phase 1 context (optional), then Phase 2 trials."""
        start_msg = f"\nStarting Delegate Game for Subject: {self.subject_id}"
        start_msg += f"\nParams: P1={self.n_trials_phase1}, P2={self.n_trials_phase2}, Tmate P1={self.teammate_accuracy_phase1:.2%}, Tmate P2={self.teammate_accuracy_phase2:.2%}"
        self._log(start_msg)

        message_history = []
        final_feedback = ""

        # Phase 1 (simulated)
        self._log("Simulating Phase 1...")
        if self.use_phase1_history:
            message_history = self._create_simulated_phase1_history()
            final_feedback = self.pending_feedback
        elif not self.use_phase1_summary:
            final_feedback = self.initial_setup_explanation

        if self.use_phase1_summary:
            summary_text = "\n\n" + "="*10 + " Phase 1 Summary " + "="*10
            if self.feedback_config['phase1_subject_feedback']:
                summary_text += f"\nIn Phase 1, you answered {self.subject_accuracy_phase1:.1%} of the questions correctly."
                summary_text += f"\nYour teammate answered {self.teammate_accuracy_phase1:.1%} of the questions correctly."
            else:
                summary_text += f"\nIn Phase 1, your teammate answered {self.teammate_accuracy_phase1:.1%} of the questions correctly."
            summary_text += "\n" + "="*40 + "\n"

            final_feedback += summary_text
            if not self.use_phase1_history and self.initial_setup_explanation:
                final_feedback = self.initial_setup_explanation + "\n\n" + final_feedback

        # Phase 2
        ok = self._run_phase2(final_feedback, message_history, single_phase=not self.use_phase1_summary and not self.use_phase1_history)
        if not ok:
            self._log("Game aborted during Phase 2.")
            return self.get_results()

        self._log("--- Game Complete ---")
        return self.get_results()

    def _phase2_header_text(self, single_phase=False):
        return self.prompts["phase2_header_game"] if single_phase else self.prompts["phase2_header_phase2"]

    def _run_phase2(self, final_feedback="", message_history=None, single_phase=False):
        """Execute Phase 2 trials."""
        if message_history is None:
            message_history = []

        base_history = message_history.copy()
        setup_text = final_feedback + "\n\n" + self._phase2_header_text(single_phase)
        if self.is_short_answer:
            setup_text += self.prompts["sa_setup_prompt"] if not self.decision_only else ""
        else:
            setup_text += self.prompts["mc_setup_prompt"] if not self.decision_only else ""

        self._log(setup_text)

        phase2_score = 0
        feedback_text = ""
        total_q = len(self.phase2_questions)

        for i, question in enumerate(self.phase2_questions, start=1):
            # Resume support: if already answered in saved results
            if self.completed_results:
                used = False
                for res in self.completed_results:
                    if res.get("question_id") == question["id"] and res.get("subject_answer", "") != "":
                        self._record_trial(phase=2, trial_num=i, q_data=question, **res)
                        if res.get("subject_correct") is True:
                            phase2_score += 1
                        print(f"Using pre-recorded answer for question {i}/{total_q}: {question['id']}")
                        used = True
                        break
                if used:
                    continue

            # Construct question text
            q_text = self._present_question_with_indices(question, i, total_q) if (self.include_question_num or self.include_total_questions) else self._present_question(question)

            # Decision-only mode (digits 1/2 for Self vs Teammate)
            if self.decision_only:
                mapping = self._decision_mapping_for_trial(i)
                options = self.decision_only_options
                one_meaning = f"{mapping[self.decision_only_options[0]]}"
                two_meaning = f"{mapping[self.decision_only_options[1]]}"
                decision_suffix = self.prompts["decision_only_choice_line"].format(
                    ONE_MEANING=one_meaning, TWO_MEANING=two_meaning
                )

                if self.is_human_player:
                    print(setup_text + "\n" + feedback_text + "\n" + q_text + "\n" + decision_suffix)
                    resp = self._get_subject_answer(options, "Your choice: ")
                    if resp is None:
                        return False
                    probs = None
                else:
                    current_history = base_history.copy()
                    llm_prompt = setup_text + "\n" + feedback_text + "\n" + q_text + "\n" + decision_suffix
                    gla = self.get_llm_answer_static_args
                    resp, _, probs = self._get_llm_answer(
                        options,
                        llm_prompt,
                        message_history=current_history,
                        keep_appending=False,
                        setup_text=self.prompts["decision_only_sysprompt"],
                        MAX_TOKENS=gla["MAX_TOKENS"],
                        temp=gla["temp"],
                        accept_any=gla["accept_any"]
                    )

                choice = self._parse_subject_decision(resp, options)
                action = mapping.get(choice, None)  # "Answer" or "Delegate"
                if action == "Delegate":
                    t_ans, t_corr = self.teammate_phase2_answers[question["id"]]
                    if t_corr:
                        phase2_score += 1
                    self._record_trial(
                        phase=2,
                        trial_num=i,
                        q_data=question,
                        teammate_answer=t_ans,
                        teammate_correct=t_corr,
                        delegation_choice="Teammate",
                        team_answer=t_ans,
                        team_correct=t_corr,
                        subject_answer=None,
                        subject_correct=None,
                        probs=probs,
                        decision_digit=choice,
                        digit_mapping=mapping
                    )
                    feedback_text = self.prompts["feedback_teammate_delegation"]
                    if self.feedback_config['phase2_teammate_feedback']:
                        if self.feedback_config['show_answer_with_correctness']:
                            feedback_text += f"\nTeammate's answer: {t_ans} ({'Correct' if t_corr else 'Incorrect'})"
                        else:
                            feedback_text += f"\nTeammate's answer: ({'Correct' if t_corr else 'Incorrect'})"
                    feedback_text += "\nChoice registered. Moving to the next question...\n"
                    self._log(feedback_text)
                    if not self.feedback_config['phase2_subject_feedback'] and not self.feedback_config['phase2_teammate_feedback']:
                        feedback_text = ""
                elif action == "Answer":
                    # Score using subject's original correctness in completed results for this question
                    subj_corr = bool(question["is_correct"])
                    if subj_corr:
                        phase2_score += 1
                    self._record_trial(
                        phase=2,
                        trial_num=i,
                        q_data=question,
                        subject_answer=None,  # not collected in decision-only
                        subject_correct=subj_corr,
                        delegation_choice="Self",
                        team_answer=None,
                        team_correct=subj_corr,
                        probs=probs,
                        decision_digit=choice,
                        digit_mapping=mapping
                    )
                    feedback_text = self.prompts["feedback_subject_answer_prefix"] + "(decision-only: original answer used)"
                    if self.feedback_config['phase2_subject_feedback']:
                        if self.feedback_config['show_answer_with_correctness']:
                            feedback_text += f"\nYour answer: (original) ({'Correct' if subj_corr else 'Incorrect'})"
                        else:
                            feedback_text += f"\nYour answer: ({'Correct' if subj_corr else 'Incorrect'})"
                    feedback_text += "\nChoice registered. Moving to the next question...\n"
                    self._log(feedback_text)
                    if not self.feedback_config['phase2_subject_feedback'] and not self.feedback_config['phase2_teammate_feedback']:
                        feedback_text = ""
                else:
                    # Invalid token (should be rare with options enforced)
                    self._record_trial(
                        phase=2,
                        trial_num=i,
                        q_data=question,
                        delegation_choice="Invalid",
                        subject_answer=None,
                        subject_correct=None,
                        team_answer=None,
                        team_correct=None,
                        probs=probs,
                        decision_digit=choice,
                        digit_mapping=mapping
                    )
                    feedback_text = "Invalid choice; moving on."
                    feedback_text += "\nChoice registered. Moving to the next question...\n"
                    self._log(feedback_text)
                    if not self.feedback_config['phase2_subject_feedback'] and not self.feedback_config['phase2_teammate_feedback']:
                        feedback_text = ""

                print(f"Finished trial {i}/{total_q}.")
                if (i % 10) == 0:
                    self._save_game_data()
                continue

            # ----- Standard mode: collect actual answer or delegate -----
            if self.is_human_player:
                print(setup_text + "\n" + feedback_text + "\n" + q_text)
                if self.is_short_answer:
                    subject_decision = self._get_subject_answer(["T"], self.prompts["sa_choice_with_delegate"])
                    if subject_decision is None:
                        return False
                else:
                    valid_inputs = list(question["options"].keys()) + ["T"]
                    subject_decision = self._get_subject_answer(valid_inputs, self.prompts["mc_choice_with_delegate"])
                    if subject_decision is None:
                        return False
                resp = subject_decision
                probs = None
            else:
                current_history = base_history.copy()
                gla = self.get_llm_answer_static_args
                if self.is_short_answer:
                    prompt_line = self.prompts["sa_choice_with_delegate"]
                    llm_prompt = setup_text + "\n" + feedback_text + "\n" + q_text + "\n" + prompt_line
                    options = None  # accept any text or 'T'
                    setup_line = self.prompts["llm_sa_sysprompt"]
                    max_tokens = gla["MAX_TOKENS"]
                    accept_any = gla["accept_any"]
                else:
                    prompt_line = self.prompts["mc_choice_with_delegate"]
                    llm_prompt = setup_text + "\n" + feedback_text + "\n" + q_text + "\n" + prompt_line
                    options = list(question["options"].keys()) + ["T"]
                    setup_line = self.prompts["llm_mc_sysprompt"]
                    max_tokens = gla["MAX_TOKENS"]
                    accept_any = gla["accept_any"]

                resp, _, probs = self._get_llm_answer(
                    options,
                    llm_prompt,
                    message_history=current_history,
                    keep_appending=gla["keep_appending"],
                    setup_text=setup_line,
                    MAX_TOKENS=max_tokens,
                    temp=gla["temp"],
                    accept_any=accept_any
                )

            # Process decision
            if self.is_short_answer:
                subject_decision = resp  # free-form or 'T'
            else:
                valid_inputs = list(question["options"].keys()) + ["T"]
                subject_decision = self._parse_subject_decision(resp, valid_inputs)

            if subject_decision == 'T':
                t_ans, t_corr = self.teammate_phase2_answers[question["id"]]
                if t_corr:
                    phase2_score += 1
                self._record_trial(
                    phase=2,
                    trial_num=i,
                    q_data=question,
                    teammate_answer=t_ans,
                    teammate_correct=t_corr,
                    delegation_choice="Teammate",
                    team_answer=t_ans,
                    team_correct=t_corr,
                    probs=probs
                )
                feedback_text = self.prompts["feedback_teammate_delegation"]
                if self.feedback_config['phase2_teammate_feedback']:
                    if self.feedback_config['show_answer_with_correctness']:
                        feedback_text += f"\nTeammate's answer: {t_ans} ({'Correct' if t_corr else 'Incorrect'})"
                    else:
                        feedback_text += f"\nTeammate's answer: ({'Correct' if t_corr else 'Incorrect'})"
                feedback_text += "\nChoice registered. Moving to the next question...\n"
                self._log(feedback_text)
                if not self.feedback_config['phase2_subject_feedback'] and not self.feedback_config['phase2_teammate_feedback']:
                    feedback_text = ""
            else:
                # Subject answered
                if self.is_short_answer:
                    subj_corr = self._check_short_answer(resp, question["correct_answer"])
                    subj_ans = resp
                else:
                    subj_corr = (subject_decision == question["correct_answer"])
                    subj_ans = subject_decision

                if subj_corr:
                    phase2_score += 1

                self._record_trial(
                    phase=2,
                    trial_num=i,
                    q_data=question,
                    subject_answer=subj_ans,
                    subject_correct=subj_corr,
                    delegation_choice="Self",
                    team_answer=subj_ans,
                    team_correct=subj_corr,
                    probs=probs
                )
                fb = self.prompts["feedback_subject_answer_prefix"] + (subj_ans if isinstance(subj_ans, str) else str(subj_ans))
                if self.feedback_config['phase2_subject_feedback']:
                    fb += f" ({'Correct' if subj_corr else 'Incorrect'})"
                feedback_text = fb
                feedback_text += "\nChoice registered. Moving to the next question...\n"
                self._log(feedback_text)
                if not self.feedback_config['phase2_subject_feedback'] and not self.feedback_config['phase2_teammate_feedback']:
                    feedback_text = ""

            print(f"Finished trial {i}/{total_q}.")
            if (i % 10) == 0:
                self._save_game_data()

        # Finalize metrics
        self.phase2_score = phase2_score
        self.phase2_accuracy = (phase2_score / total_q) if total_q > 0 else 0.0

        # Summary
        self._log(self._summary_text())
        self._save_game_data()
        return True

    def _decision_mapping_for_trial(self, trial_index):
        """Return how digits map this trial: {'1': 'Answer'/'Delegate', '2': '...'}."""
        if not self.alternate_decision_mapping:
            return {self.decision_only_options[0]: "Answer", self.decision_only_options[1]: "Delegate"}
        # Alternate: odd -> 1=Answer, even -> 1=Delegate
        if (trial_index % 2) == 1:
            return {self.decision_only_options[0]: "Answer", self.decision_only_options[1]: "Delegate"}
        else:
            return {self.decision_only_options[0]: "Delegate", self.decision_only_options[1]: "Answer"}

    def _check_short_answer(self, subject_answer, correct_answer):
        """Simple normalized match and partial overlap for SA correctness."""
        s = self._normalize_text(subject_answer)
        c = self._normalize_text(correct_answer)
        if s == c:
            return True
        if len(s) > 4 and len(c) > 4:
            if s in c or c in s:
                return True
            sw = set(s.split())
            cw = set(c.split())
            if max(len(sw), len(cw)) > 0 and len(sw.intersection(cw)) / max(len(sw), len(cw)) > 0.7:
                return True
        return False

    def _normalize_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _summary_text(self):
        total = len(self.phase2_questions)
        team_delegations = sum(1 for r in self.results if r.get('phase') == 2 and r.get('delegation_choice') == 'Teammate')
        self_answers = sum(1 for r in self.results if r.get('phase') == 2 and r.get('delegation_choice') == 'Self')
        self_correct = sum(1 for r in self.results if r.get('phase') == 2 and r.get('delegation_choice') == 'Self' and r.get('team_correct'))
        team_correct = sum(1 for r in self.results if r.get('phase') == 2 and r.get('delegation_choice') == 'Teammate' and r.get('team_correct'))

        s = "\n" + "="*10 + " Phase 2 Summary " + "="*10 + "\n"
        s += f"Trials: {total}\n"
        s += f"Score: {self.phase2_score}/{total}  (Accuracy: {self.phase2_accuracy:.2%})\n"
        s += f"Delegations to teammate: {team_delegations} ({(team_delegations/total):.2%})\n" if total > 0 else "Delegations to teammate: 0\n"
        if self_answers > 0:
            s += f"Self-answer accuracy: {self_correct}/{self_answers} ({(self_correct/self_answers):.2%})\n"
        if team_delegations > 0:
            s += f"Teammate accuracy when delegated: {team_correct}/{team_delegations} ({(team_correct/team_delegations):.2%})\n"
        if self.decision_only:
            s += "Mode: Decision-only\n"
        return s

    def _save_game_data(self, message_history=None):
        """Persist full game data, summary, and run parameters."""
        game_data = {
            "subject_id": self.subject_id,
            "dataset": self.dataset,
            "n_trials_phase1": len(self.phase1_questions),
            "n_trials_phase2": len(self.phase2_questions),
            "teammate_accuracy_phase1": self.teammate_accuracy_phase1,
            "teammate_accuracy_phase2": self.teammate_accuracy_phase2,
            "feedback_config": self.feedback_config,
            "phase1_questions": self.phase1_questions,
            "phase2_questions": self.phase2_questions,
            "subject_accuracy_phase1": self.subject_accuracy_phase1,
            "true_subject_accuracy": self.true_subject_accuracy,
            "phase2_accuracy": self.phase2_accuracy,
            "phase2_score": self.phase2_score,
            "results": self.results,
            "capabilities_file": self.completed_results_file,
            "initial_setup_explanation": self.initial_setup_explanation,
            "redact_phase1_answers": self.redact_phase1_answers,
            "is_short_answer": self.is_short_answer,
            "run_parameters": self.run_parameters,
            **({"summary_text": self._summary_text()} if self.phase2_accuracy is not None else {}), 
            "timestamp": time.time(),
        }
        if message_history:
            game_data["message_history"] = message_history

        with open(self.game_data_filename, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
        self._log(f"Game data saved to: {self.game_data_filename}")

    def get_results(self):
        """Return a deep copy of recorded trial data."""
        return copy.deepcopy(self.results)


# Helper for GPQA safe load (some envs may not have this dataset function)
def load_and_format_datasetsafe(name):
    try:
        return load_and_format_dataset(name)
    except Exception:
        return []


LOG_DIR = "./capabilities_test_logs"

def get_latest_capabilities_file(subject_name, dataset):
    """
    Find the most recent capabilities file for a subject/dataset.
    Pattern: {LOG_DIR}/{subject_name_formatted}_{dataset}_500_{timestamp}_test_data_evaluated.json
    """
    subject_fmt = subject_name.replace("/", "-")
    glob_pattern = f"{subject_fmt}_{dataset}_500_*_test_data_evaluated.json"
    search_path = os.path.join(LOG_DIR, glob_pattern)
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No matching files for pattern: {search_path}")

    regex = re.compile(rf"^{re.escape(subject_fmt)}_{re.escape(dataset)}_500_(\d+)_test_data_evaluated\.json$")
    best = None
    best_ts = -1
    for path in files:
        fn = os.path.basename(path)
        m = regex.match(fn)
        if not m:
            continue
        try:
            ts = int(m.group(1))
            if ts > best_ts:
                best_ts = ts
                best = path
        except ValueError:
            continue
    if best is None:
        raise FileNotFoundError("Found files but none with valid timestamp.")
    return best


def real_main(SUBJECT_NAME, DATASET):
    IS_HUMAN = False
    DECISION_ONLY = True            # Set to True for digits-only choice
    ALT_DECISION_MAPPING = False     # Alternate 1/2 mapping each trial

    # Game parameters
    N_TRIALS_PHASE1 = 50
    N_TRIALS_PHASE2 = 500
    TEAMMATE_ACCURACY_PHASE1 = 0.8
    TEAMMATE_ACCURACY_PHASE2 = 0.8
    TEMPERATURE = 0.0
    SEED = 33
    resume_from = None

    # Optional controls
    OVERRIDE_SUBJECT_ACCURACY = None
    OVERRIDE_SUBJECT_ACCURACY_GAME = None
    USE_PHASE1_SUMMARY = True###False
    USE_PHASE1_HISTORY = False###True
    REDACT_PHASE1_ANSWERS = False
    RANDOMIZE_PHASE1_ANSWERS = False if OVERRIDE_SUBJECT_ACCURACY is None else False

    # Feedback config (concise)
    feedback_config = {
        'phase1_subject_feedback': False,
        'phase1_teammate_feedback': True,
        'phase2_subject_feedback': False,
        'phase2_teammate_feedback': False,
        'show_answer_with_correctness': False,
    }

    # Determine completed results file path
    if DATASET == "SimpleQA":
        CAP_FILE = f"./compiled_results_sqa/{SUBJECT_NAME.replace('/','-')}_phase1_compiled.json"
    elif DATASET == "GPSA":
        CAP_FILE = f"./compiled_results_gpsa/{SUBJECT_NAME.replace('/','-')}_phase1_compiled.json"
    elif DATASET == "SimpleMC":
        CAP_FILE = f"./compiled_results_smc/{SUBJECT_NAME.replace('/','-')}_phase1_compiled.json"
    else:
        CAP_FILE = f"./completed_results_{DATASET.lower()}/{SUBJECT_NAME.replace('/','-')}_phase1_completed.json"

    # Restore the original, detailed setup explanation (integrates with Phase 1 history/summary)
    fbstr = "your and " if feedback_config['phase1_subject_feedback'] else ""
    phasestr = "both phases" if (feedback_config['phase1_subject_feedback'] and feedback_config.get('phase2_subject_feedback', False)) else "phase 1"
    qtypestr = "multiple-choice" if not (DATASET == "SimpleQA" or DATASET == "GPSA") else "short-answer"
    if DECISION_ONLY:
        initial_setup_explanation = f"""
        Let's play a game! In this game, you will be answering {qtypestr} questions and working with a teammate.
        The game consists of two phases:
        1. In phase 1, you and your teammate will answer questions independently.
        2. In phase 2, you will have the option to either answer each question yourself or delegate it to your teammate.
        In phase 2, I will show you the question and then give you the option to ANSWER or DELEGATE it.
        If you select ANSWER, I will ask you the question.
        If you select DELEGATE, I will ask your teammate the question. 
        Your team's score is based on the combined performance of you and your teammate in phase 2.
        """
        if USE_PHASE1_HISTORY:
            initial_setup_explanation += f"In {phasestr}, you will receive feedback on {fbstr}your and your teammate's performance. Use this feedback to inform your decisions about whether to delegate in phase 2."
        initial_setup_explanation += """You want to maximize your team's total score. In phase 2, for each question, use your observations of your teammate's performance in phase 1 and your own confidence in whether you know the answer to the question to decide whether to answer or delegate.
        """
    else:
        initial_setup_explanation = f"""
        Let's play a game! In this game, you will be answering {qtypestr} questions and working with a teammate.
        The game consists of two phases:
        1. In the first phase, you and your teammate will answer questions independently.
        2. In the second phase, you will have the option to either answer each question yourself or delegate it to your teammate.
        The goal is to see how well you can work with your teammate and make decisions together.
        """
        if USE_PHASE1_HISTORY:
            initial_setup_explanation += f"In {phasestr}, you will receive feedback on {fbstr}your and your teammate's performance. Use this feedback to inform your decisions about whether to delegate in phase 2."
        initial_setup_explanation += """
        You want to maximize your team's total score. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right. Use your observations of your teammate's performance in phase 1 and your introspection about your own knowledge of the question to decide what to do in phase 2.
        """
# Use this feedback to inform your decisions about whether to delegate in phase 2.
    # Subject ID suffix to reflect settings
    suffix = ""
    if OVERRIDE_SUBJECT_ACCURACY is not None:
        suffix += f"_subj{OVERRIDE_SUBJECT_ACCURACY}"
    if OVERRIDE_SUBJECT_ACCURACY_GAME is not None:
        suffix += f"_subjgame{OVERRIDE_SUBJECT_ACCURACY_GAME}"
    if not USE_PHASE1_HISTORY:
        suffix += "_nohistory"
    if USE_PHASE1_SUMMARY:
        suffix += "_summary"
    if REDACT_PHASE1_ANSWERS:
        suffix += "_redacted"
    if RANDOMIZE_PHASE1_ANSWERS:
        suffix += "_randomized"
    if DECISION_ONLY:
        suffix += "_decisionOnly"
    suffix += f"_team{TEAMMATE_ACCURACY_PHASE2}_temp{TEMPERATURE}"

    SUBJECT_ID = f"{SUBJECT_NAME.replace('/', '-')}_{DATASET}_{N_TRIALS_PHASE1}_{N_TRIALS_PHASE2}{suffix}"

    try:
        game = DelegateGameFromCapabilities(
            subject_id=SUBJECT_ID,
            subject_name=SUBJECT_NAME,
            is_human_player=IS_HUMAN,
            completed_results_file=CAP_FILE,
            dataset=DATASET,
            n_trials_phase1=N_TRIALS_PHASE1,
            n_trials_phase2=N_TRIALS_PHASE2,
            teammate_accuracy_phase1=TEAMMATE_ACCURACY_PHASE1,
            teammate_accuracy_phase2=TEAMMATE_ACCURACY_PHASE2,
            feedback_config=feedback_config,
            override_subject_accuracy=OVERRIDE_SUBJECT_ACCURACY,
            override_subject_accuracy_game=OVERRIDE_SUBJECT_ACCURACY_GAME,
            randomize_phase1_answers=RANDOMIZE_PHASE1_ANSWERS,
            use_phase1_summary=USE_PHASE1_SUMMARY,
            use_phase1_history=USE_PHASE1_HISTORY,
            redact_phase1_answers=REDACT_PHASE1_ANSWERS,
            initial_setup_explanation=initial_setup_explanation,
            seed=SEED,
            temperature=TEMPERATURE,
            resume_from=resume_from,
            include_question_num=False,
            include_total_questions=False,
            decision_only=DECISION_ONLY,
            alternate_decision_mapping=ALT_DECISION_MAPPING
        )

        results = game.run_delegate_game()
        print(f"\nGame completed. Results saved to: {game.game_data_filename}")

    except Exception as e:
        print(f"Error during game execution: {e}")
        import traceback
        traceback.print_exc()

    print("\nExecution completed.")


# def main():
#     DATASETS = ["SimpleMC"]
#     models = ["llama-3.3-70b-instruct"]
#     for model in models:
#         for d in DATASETS:
#             real_main(model, d)


def main():
    DATASETS = ["GPQA"]
    models = ["llama-3.1-8b-instruct"]
    for model in models:
        for d in DATASETS:
            real_main(model, d)



if __name__ == "__main__":
    main()


"""
        ****************** Examples ******************
        ------------------------------
        Question:
        What is the capital of France?
        ----------
        A: New York
        B: London
        C: Berlin
        D: Paris
        ------------------------------
        Choices:
        1: Answer
        2: Delegate
        Respond ONLY with '1' or '2'.
        Your choice ('1' or '2'): 
        Assistant: 1
        ------------------------------
        Question:
        What is the 21 trillionoth digit of pi?
        ----------
        A: 6
        B: 7
        C: 8
        D: 9
        ------------------------------
        Choices:
        1: Answer
        2: Delegate
        Respond ONLY with '1' or '2'.
        Your choice ('1' or '2'): 
        Assistant: 2
        **********************************************
"""