import time
import json
from load_and_format_datasets import load_and_format_dataset
from base_game_class import *
import random
import string

class CapabilitiesTest(BaseGameClass):
    """
    Just ask independent multiple-choice or short answer questions and record responses.
    """
    def __init__(
        self,
        subject_id,
        subject_name,
        questions,
        n_questions=None,
        is_human_player=False,
        resume_from=None,
        temperature=0.0,
        resample_for_probs=False,
        nested=None,
        include_question_num=False,
        include_total_questions=False,
        explicit_confidence_task=False
    ):
        """
        Args:
            subject_id (str): Identifier for the subject/session
            subject_name (str): Name of the subject (model name for LLMs)
            questions (list): Formatted questions to use
            n_questions (int): How many questions to use
            is_human_player (bool): Whether the subject is a human player or an LLM
            resume_from (string): Filename to resume from (in case game got interrupted)
            include_question_num (bool): If True, pass question_num to present_question
            include_total_questions (bool): If True, pass total_questions to present_question
            explicit_confidence_task (bool): If True, ask two separate prompts: first the question directly (for logprobs), 
                then the confidence question. The LLM won't know it already answered the question.
                Note: This parameter is ignored when nested="All" (which already does separate queries).
        """
        if explicit_confidence_task:
            filepath = "explicit_confidence_task_logs"
        else:
            filepath = "capabilities_test_logs" if not nested else "capabilities_3p_test_logs" if nested == "Other" else "capabilities_1p_test_logs"
        super().__init__(subject_id, subject_name, is_human_player, filepath)
        self.n_questions = len(questions) if not n_questions else n_questions

        # Set up state variables
        self.results = {}
        self.questions = []
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = None
        self.temperature = temperature
        self.log_suffix = ""  # Will be set in main() based on nested parameter
        self.resample_for_probs = resample_for_probs
        self.nested = nested

        # Control passing indices to present_question
        self.include_question_num = include_question_num
        self.include_total_questions = include_total_questions
        self.explicit_confidence_task = explicit_confidence_task

        self.mc_setup_prompt = "I'm going to ask you a series of multiple-choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text."
        self.sa_setup_prompt = "I'm going to ask you a series of short-answer questions. For each one, respond as succinctly as possible. Answer as best you can, even if you're not certain."
        self.human_mc_input_prompt = "Your answer (A, B, C, or D): "
        self.human_sa_input_prompt = "Your answer: "

        # Explicit likelihood self/other
        self.nested_question_prompt = None
        self.nested_option_dict = None
        self.nested_range_midpoints = None
        if self.nested == "Other":
            self.nested_option_dict = {"A": "<5%", "B": "5-10%", "C": "10-20%", "D": "20-40%", "E": "40-60%", "F": "60-80%", "G": "80-90%", "H": ">90%"}
            self.nested_range_midpoints = {"A": 0.025, "B": 0.075, "C": 0.15,  "D": 0.3, "E": 0.5, "F": 0.7, "G": 0.85, "H": 0.95}
            self.nested_setup_prompt = "I want your help calibrating question difficulty. I'm going to show you a multiple-choice question, and I want you to tell me approximately what percentage of college-educated people you think would get it right. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
            self.nested_question_prompt = "What percentage of college-educated people would get this question right?"
        elif self.nested == "Self":
            self.nested_option_dict = {"A": "<5%", "B": "5-10%", "C": "10-20%", "D": "20-40%", "E": "40-60%", "F": "60-80%", "G": "80-90%", "H": ">90%"}
            self.nested_range_midpoints = {"A": 0.025, "B": 0.075, "C": 0.15,  "D": 0.3, "E": 0.5, "F": 0.7, "G": 0.85, "H": 0.95}
            self.nested_setup_prompt = "I'm going to show you a multiple-choice question, and I want you to tell me your level of confidence that you would get the question right. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
            self.nested_question_prompt = "How confident are you that you would get this question right?"
        elif self.nested == "All":
            # For "All", we'll set up prompts dynamically when needed
            # But initialize the option dict and range midpoints for consistency
            self.nested_option_dict = {"A": "<5%", "B": "5-10%", "C": "10-20%", "D": "20-40%", "E": "40-60%", "F": "60-80%", "G": "80-90%", "H": ">90%"}
            self.nested_range_midpoints = {"A": 0.025, "B": 0.075, "C": 0.15,  "D": 0.3, "E": 0.5, "F": 0.7, "G": 0.85, "H": 0.95}

        # Run parameters for reproducibility (run-level only; no per-question values)
        self.run_parameters = {
            "n_questions": self.n_questions,
            "temperature": self.temperature,
            "resample_for_probs": self.resample_for_probs,
            "is_human_player": self.is_human_player,
            "nested": self.nested,
            "explicit_confidence_task": self.explicit_confidence_task,
            "present_question_args": {
                "include_question_num": self.include_question_num,
                "include_total_questions": self.include_total_questions
            }
            # Added during the run when applicable:
            # "parallel_config": {...}
            # "get_llm_answer_static_args": {...}
            # "mc_setup_prompt": "..."
            # "sa_setup_prompt": "..."
            # "nested_option_dict": {...}
            # "nested_range_midpoints": {...}
            # "nested_question_prompt": "..."
            # "human_mc_input_prompt": "..."
            # "human_sa_input_prompt": "..."
            # "seed": <int>  # set in main
        }

        if len(questions) < self.n_questions:
            raise ValueError(f"Not enough questions provided ({len(questions)}); ({self.n_questions} needed)")
        
        # Take the first n_questions
        self.questions = questions[:self.n_questions]
        self._log(f"Using {len(self.questions)} provided questions")

        if resume_from and resume_from != "":
            try:
                with open(resume_from, "r") as f:
                    prev_data = json.load(f)
            except Exception as e:
                self._log(f"ERROR: Error opening resume file: {str(e)}")
                return False
            self.results = prev_data["results"]
            self._log(f"Resuming from {resume_from} holding {len(self.results)} questions")
            for rdict in self.results.values():
                if rdict["is_correct"] == True: self.correct_count +=1
                self.total_count += 1
            self.questions = [q for q in self.questions if q["id"] not in self.results]

    def _save_data(self):
        """Save data to file"""
        data = {
            "subject_id": self.subject_id,
            "timestamp": time.time(),
            "accuracy": self.accuracy,
            "results": self.results,
            "run_parameters": self.run_parameters,
        }
                    
        filename = f"{self.log_base_name}{self.log_suffix}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._log(f"Data saved to: {filename}")

    def _parse_subject_decision(self, subject_answer, options):
        """Normalize free-form subject answer to a single-letter/choice decision when possible."""
        if len(subject_answer.rstrip(string.whitespace + string.punctuation)) == 0:
            return subject_answer
        arr = subject_answer.upper().rstrip(string.whitespace + string.punctuation)
        if arr and arr[0] in options:
            return arr[0]
        if arr and arr[-1] in options:
            return arr[-1]
        return subject_answer

    def _present_question_with_indices(self, question, i, total):
        """Helper to call present_question with the configured indices."""
        if self.include_question_num and self.include_total_questions:
            return self._present_question(question, i, total)
        elif self.include_question_num:
            return self._present_question(question, i)
        else:
            return self._present_question(question)

    def _prepare_mc_for_llm(self, question, question_num=None, total_questions=None):
        """
        Prepare MC question text, setup prompt, options list, and (if nested) midpoint map.
        Uses present_question indices based on provided question_num/total_questions.
        """
        if self.nested:
            q_text = self._present_nested_question(question, self.nested_question_prompt, self.nested_option_dict)
            options = list(self.nested_option_dict.keys())
            setup_prompt = self.nested_setup_prompt
            RANGE_MIDPOINTS = self.nested_range_midpoints
        else:
            if question_num is None and total_questions is None:
                q_text = self._present_question(question)
            elif total_questions is None:
                q_text = self._present_question(question, question_num)
            else:
                q_text = self._present_question(question, question_num, total_questions)
            options = list(question["options"].keys())
            setup_prompt = self.mc_setup_prompt
            RANGE_MIDPOINTS = None

        options_str = " or ".join(options) if len(options) == 2 else ", ".join(options[:-1]) + f", or {options[-1]}"
        llm_prompt = q_text + f"\nYour choice ({options_str}): "
        return q_text, setup_prompt, options, RANGE_MIDPOINTS, llm_prompt

    def run_capabilities_measurement(self):
        """
        Measures a subject's performance on multiple choice questions.
        Uses parallel execution for resampling if configured.
        
        Returns:
            bool: True if completed successfully, False otherwise
            str: Path to the capabilities data file
        """
        start_message = f"\nStarting Capabilities Measurement for Subject: {self.subject_id}"
        self._log(start_message)
        self._log(f"Configuration: Questions={self.n_questions}, is_human_player={self.is_human_player}, temperature={self.temperature}, resample_for_probs={self.resample_for_probs}, nested={self.nested}, explicit_confidence_task={self.explicit_confidence_task}")
        self._log("\n" + "="*10 + " Starting Capability Measuring " + "="*10)
        
        log_interval = 10

        # For "All", use mc_setup_prompt since the direct question uses it
        # For other nested modes, use nested_setup_prompt
        if self.nested == "All":
            self.run_parameters["setup_prompt"] = self.mc_setup_prompt
        else:
            self.run_parameters["setup_prompt"] = self.mc_setup_prompt if not self.nested else self.nested_setup_prompt
        if self.nested and self.nested != "All":
            self.run_parameters["nested_option_dict"] = self.nested_option_dict
            self.run_parameters["nested_range_midpoints"] = self.nested_range_midpoints
            self.run_parameters["nested_question_prompt"] = self.nested_question_prompt
        elif self.nested == "All":
            # For "All", record all the nested parameters since we use them for different queries
            self.run_parameters["nested_option_dict"] = self.nested_option_dict
            self.run_parameters["nested_range_midpoints"] = self.nested_range_midpoints
            # Note: nested_question_prompt varies by query type, so we don't set a single one
        # This condition diverts the logic to the parallel path
        if self.resample_for_probs and not self.is_human_player:
            #################################################################
            # PARALLEL PATH: For resampling LLM multiple-choice questions
            #################################################################
            max_workers = 4
            epsilon = 0.05
            self.run_parameters["parallel_config"] = {"max_workers": max_workers, "epsilon": epsilon}

            # --- Phase 1: Prepare all tasks ---
            self._log(f"Preparing {len(self.questions)} questions for parallel resampling...")
            estimation_tasks = []
            total_q = len(self.questions)
            for idx, question in enumerate(self.questions, start=1):
                _, setup_prompt, options, RANGE_MIDPOINTS, llm_prompt = self._prepare_mc_for_llm(
                    question,
                    idx if self.include_question_num else None,
                    total_q if self.include_total_questions else None
                )

                task = {
                    "question_obj": question,
                    "prompt": setup_prompt + "\n\n" + llm_prompt,
                    "options": options,
                    "message_history": [], # no history
                    "epsilon": epsilon,
                    "range_midpoints": RANGE_MIDPOINTS,
                }
                estimation_tasks.append(task)
            
            # --- Phase 2: Execute all tasks in parallel ---
            parallel_results = self.run_estimations_in_parallel(estimation_tasks, max_workers=max_workers)

            # --- Phase 3: Process the results ---
            self._log("Processing results from parallel execution...")
            for result_item in parallel_results:
                if result_item.get('error'):
                    self._log(f"ERROR: Task for question '{result_item['task']['question_obj'].get('id')}' failed: {result_item['error']}")
                    continue
                
                subject_answer, _, probs = result_item['result']
                question = result_item['task']['question_obj']
                options = result_item['task']['options']
                RANGE_MIDPOINTS = result_item['task'].get('range_midpoints')
                
                subject_decision = self._parse_subject_decision(subject_answer, options)

                if self.nested:
                    if probs and RANGE_MIDPOINTS:
                        is_correct = sum(
                            RANGE_MIDPOINTS[key.strip()] * mass
                            for key, mass in probs.items()
                            if key.strip() in RANGE_MIDPOINTS
                        )
                    else:
                        is_correct = 0.0
                else:
                    is_correct = (subject_decision == question["correct_answer"])

                if is_correct:
                    self.correct_count += 1
                
                if subject_decision != "":
                    self.results[question["id"]] = {
                        "question": question,
                        "subject_answer": subject_decision,
                        "is_correct": is_correct,
                        "probs": probs 
                    }
                self.total_count += 1
            
            # Save data once at the end of processing
            self._save_data()

        else:
            #################################################################
            # SEQUENTIAL PATH: For humans or single-sample runs
            #################################################################
            probs = None

            if self.is_human_player:
                # Record human input prompt used
                self.run_parameters["human_mc_input_prompt"] = self.human_mc_input_prompt
            else:
                # Record static _get_llm_answer args used in this run (MC path)
                max_tokens_used = None if ('opus-4' in self.subject_name or 'sonnet-4' in self.subject_name) else 1
                self.run_parameters["get_llm_answer_static_args"] = {
                    "keep_appending": False,
                    "message_history": [],
                    "MAX_TOKENS": max_tokens_used,
                    "temp": self.temperature,
                    "accept_any": False if 'base' in self.subject_name else True
                }

            total_q = len(self.questions)
            for i, question in enumerate(self.questions, start=1):
                if self.is_human_player:
                    # Present once, honoring index config
                    q_text = self._present_question_with_indices(question, i, total_q)
                    print(q_text)
                    subject_answer = self._get_subject_answer(
                        list(question["options"].keys()), 
                        self.human_mc_input_prompt
                    )
                    if subject_answer is None:
                        return False, None
                    options = list(question["options"].keys())
                    RANGE_MIDPOINTS = None
                    probs = None
                    confidence_answer = None
                    confidence_probs = None
                else:
                    # For LLM subject
                    gla_args = self.run_parameters["get_llm_answer_static_args"]
                    
                    if self.nested == "All":
                        # Three separate prompts: direct question, self-confidence, and other-confidence
                        # All with empty message history so LLM has no knowledge of other questions
                        # Note: explicit_confidence_task parameter is ignored when nested="All"
                        
                        # First prompt: Ask the question directly (nested=None)
                        original_nested = self.nested
                        self.nested = None
                        q_text_direct, setup_prompt_direct, options_direct, _, llm_prompt_direct = self._prepare_mc_for_llm(
                            question,
                            i if self.include_question_num else None,
                            total_q if self.include_total_questions else None
                        )
                        subject_answer, _, probs = self._get_llm_answer(
                            options_direct,
                            setup_prompt_direct + "\n\n" + llm_prompt_direct,
                            [],  # Empty message history - separate conversation
                            keep_appending=gla_args["keep_appending"],
                            MAX_TOKENS=gla_args["MAX_TOKENS"],
                            temp=gla_args["temp"],
                            accept_any=gla_args["accept_any"]
                        )
                        
                        # Second prompt: Ask self-confidence question (nested="Self")
                        self.nested = "Self"
                        self.nested_setup_prompt = "I'm going to show you a multiple-choice question, and I want you to tell me your level of confidence that you would get the question right. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
                        self.nested_question_prompt = "How confident are you that you would get this question right?"
                        q_text_self, setup_prompt_self, options_self, RANGE_MIDPOINTS, llm_prompt_self = self._prepare_mc_for_llm(
                            question,
                            i if self.include_question_num else None,
                            total_q if self.include_total_questions else None
                        )
                        self_confidence_answer, _, self_confidence_probs = self._get_llm_answer(
                            options_self,
                            setup_prompt_self + "\n\n" + llm_prompt_self,
                            [],  # Empty message history - separate conversation
                            keep_appending=gla_args["keep_appending"],
                            MAX_TOKENS=gla_args["MAX_TOKENS"],
                            temp=gla_args["temp"],
                            accept_any=gla_args["accept_any"]
                        )
                        
                        # Third prompt: Ask other-confidence question (nested="Other")
                        self.nested = "Other"
                        self.nested_setup_prompt = "I want your help calibrating question difficulty. I'm going to show you a multiple-choice question, and I want you to tell me approximately what percentage of college-educated people you think would get it right. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
                        self.nested_question_prompt = "What percentage of college-educated people would get this question right?"
                        q_text_other, setup_prompt_other, options_other, _, llm_prompt_other = self._prepare_mc_for_llm(
                            question,
                            i if self.include_question_num else None,
                            total_q if self.include_total_questions else None
                        )
                        other_confidence_answer, _, other_confidence_probs = self._get_llm_answer(
                            options_other,
                            setup_prompt_other + "\n\n" + llm_prompt_other,
                            [],  # Empty message history - separate conversation
                            keep_appending=gla_args["keep_appending"],
                            MAX_TOKENS=gla_args["MAX_TOKENS"],
                            temp=gla_args["temp"],
                            accept_any=gla_args["accept_any"]
                        )
                        
                        # Restore original nested value
                        self.nested = original_nested
                        
                        # Parse decisions
                        subject_decision = self._parse_subject_decision(subject_answer, options_direct)
                        self_confidence_decision = self._parse_subject_decision(self_confidence_answer, options_self)
                        other_confidence_decision = self._parse_subject_decision(other_confidence_answer, options_other)
                        options = options_direct
                        is_correct = (subject_decision == question["correct_answer"])
                        confidence_answer = self_confidence_decision
                        confidence_probs = self_confidence_probs
                        confidence_decision = self_confidence_decision
                        
                    elif self.explicit_confidence_task:
                        # Two separate prompts: first the direct question, then the confidence question
                        # First prompt: Ask the question directly (without nested) to get logprobs
                        original_nested = self.nested
                        self.nested = None  # Temporarily disable nested for direct question
                        q_text_direct, setup_prompt_direct, options_direct, _, llm_prompt_direct = self._prepare_mc_for_llm(
                            question,
                            i if self.include_question_num else None,
                            total_q if self.include_total_questions else None
                        )
                        # Use empty message history so LLM doesn't know about previous answer
                        subject_answer, _, probs = self._get_llm_answer(
                            options_direct,
                            setup_prompt_direct + "\n\n" + llm_prompt_direct,
                            [],  # Empty message history - separate conversation
                            keep_appending=gla_args["keep_appending"],
                            MAX_TOKENS=gla_args["MAX_TOKENS"],
                            temp=gla_args["temp"],
                            accept_any=gla_args["accept_any"]
                        )
                        
                        # Second prompt: Ask the confidence question (with nested="Self")
                        self.nested = "Self"
                        # Set up nested prompts if not already set
                        if not self.nested_option_dict:
                            self.nested_option_dict = {"A": "<5%", "B": "5-10%", "C": "10-20%", "D": "20-40%", "E": "40-60%", "F": "60-80%", "G": "80-90%", "H": ">90%"}
                            self.nested_range_midpoints = {"A": 0.025, "B": 0.075, "C": 0.15,  "D": 0.3, "E": 0.5, "F": 0.7, "G": 0.85, "H": 0.95}
                            self.nested_setup_prompt = "I'm going to show you a multiple-choice question, and I want you to tell me your level of confidence that you would get the question right. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
                            self.nested_question_prompt = "How confident are you that you would get this question right?"
                        
                        q_text_confidence, setup_prompt_confidence, options_confidence, RANGE_MIDPOINTS, llm_prompt_confidence = self._prepare_mc_for_llm(
                            question,
                            i if self.include_question_num else None,
                            total_q if self.include_total_questions else None
                        )
                        # Use empty message history - separate conversation
                        confidence_answer, _, confidence_probs = self._get_llm_answer(
                            options_confidence,
                            setup_prompt_confidence + "\n\n" + llm_prompt_confidence,
                            [],  # Empty message history - separate conversation
                            keep_appending=gla_args["keep_appending"],
                            MAX_TOKENS=gla_args["MAX_TOKENS"],
                            temp=gla_args["temp"],
                            accept_any=gla_args["accept_any"]
                        )
                        # Restore original nested value
                        self.nested = original_nested
                        
                        # Use direct answer for correctness, confidence answer for comparison
                        subject_decision = self._parse_subject_decision(subject_answer, options_direct)
                        confidence_decision = self._parse_subject_decision(confidence_answer, options_confidence)
                        options = options_direct
                        is_correct = (subject_decision == question["correct_answer"])
                    else:
                        # Original behavior: single prompt
                        _, setup_prompt, options, RANGE_MIDPOINTS, llm_prompt = self._prepare_mc_for_llm(
                            question,
                            i if self.include_question_num else None,
                            total_q if self.include_total_questions else None
                        )

                        subject_answer, _, probs = self._get_llm_answer(
                            options,
                            setup_prompt + "\n\n" + llm_prompt,
                            gla_args["message_history"],
                            keep_appending=gla_args["keep_appending"],
                            MAX_TOKENS=gla_args["MAX_TOKENS"],
                            temp=gla_args["temp"],
                            accept_any=gla_args["accept_any"]
                        )
                        confidence_answer = None
                        confidence_probs = None
                        confidence_decision = None
                
                # --- Same result processing logic as parallel path ---
                if not self.explicit_confidence_task and self.nested != "All":
                    subject_decision = self._parse_subject_decision(subject_answer, options)

                    if self.nested:
                        is_correct = (sum(
                            RANGE_MIDPOINTS[key.strip()] * mass
                            for key, mass in (probs or {}).items()
                            if key.strip() in RANGE_MIDPOINTS
                        ) if probs else RANGE_MIDPOINTS[subject_decision] if (RANGE_MIDPOINTS and subject_decision in RANGE_MIDPOINTS) else 0.0)
                    else:
                        is_correct = (subject_decision == question["correct_answer"])

                if is_correct:
                    self.correct_count += 1
                
                if subject_decision != "":
                    result_dict = {
                        "question": question,
                        "subject_answer": subject_decision,
                        "is_correct": is_correct,
                        "probs": probs 
                    }
                    if self.nested == "All":
                        result_dict["self_confidence_answer"] = self_confidence_decision
                        result_dict["self_confidence_probs"] = self_confidence_probs
                        result_dict["other_confidence_answer"] = other_confidence_decision
                        result_dict["other_confidence_probs"] = other_confidence_probs
                    elif self.explicit_confidence_task:
                        result_dict["confidence_answer"] = confidence_decision
                        result_dict["confidence_probs"] = confidence_probs
                    self.results[question["id"]] = result_dict
                self.total_count += 1
                print(f"Completed question {self.total_count}/{len(self.questions)}")
                if (i) % log_interval == 0: self._save_data()
        
        # --- Finalization steps, common to both paths ---
        if self.total_count > 0:
            self.accuracy = self.correct_count / self.total_count
        else:
            self.accuracy = 0.0
            self._log("WARNING: No questions were processed.")
        
        summary = f"\nCapabilities Test Complete. Accuracy: {self.accuracy:.2%} ({self.correct_count}/{self.total_count})"
        self._log(summary)
        
        self._save_data()
                    
        capabilities_file_path = f"{self.log_base_name}{self.log_suffix}.json"
        self._log(f"Capabilities measurement completed. Results saved to: {capabilities_file_path}")
        return True, capabilities_file_path

    def run_capabilities_measurement_sa(self):
        """
        This measures a subject's performance on short answer questions and saves the results to a file.
        
        Returns:
            bool: True if completed successfully, False otherwise
            str: Path to the capabilities data file
        """
        start_message = f"\nStarting Capabilities Measurement for Subject: {self.subject_id}"
        self._log(start_message)
        self._log(f"Configuration: Questions={self.n_questions}, is_human_player={self.is_human_player}, temperature={self.temperature}, resample_for_probs={self.resample_for_probs}, nested={self.nested}")
        self._log("\n" + "="*10 + " Starting Capability Measuring " + "="*10)
        
        # Initialize state
        probs = None
        log_interval = 10
        self.accuracy = None

        # Record fixed prompts/args used for this SA run
        if self.is_human_player:
            self.run_parameters["human_sa_input_prompt"] = self.human_sa_input_prompt
        else:
            self.run_parameters["sa_setup_prompt"] = self.sa_setup_prompt
            # Use a reasonable limit for short answers instead of None (which defaults to 32000)
            max_tokens_sa = 1024  # Reasonable limit for short answers
            self.run_parameters["get_llm_answer_static_args"] = {
                "keep_appending": False,
                "message_history": [],
                "MAX_TOKENS": max_tokens_sa,
                "temp": self.temperature
            }
        
        # Process each question
        total_q = len(self.questions)
        for i, question in enumerate(self.questions, start=1):
            # Present honoring index config
            q_text = self._present_question_with_indices(question, i, total_q)

            # Get subject's answer
            if self.is_human_player:
                print(q_text)
                subject_answer = self._get_subject_answer(
                    [], 
                    self.human_sa_input_prompt
                )
                if subject_answer is None:
                    return False
                probs = None
            else:
                # For LLM subject
                llm_prompt = q_text + "\nYour answer: "
                setup_prompt = self.sa_setup_prompt
                gla_args = self.run_parameters["get_llm_answer_static_args"]
                subject_answer, _, probs = self._get_llm_answer(
                    None,
                    setup_prompt + "\n\n" + llm_prompt,
                    gla_args["message_history"], # no history
                    keep_appending=gla_args["keep_appending"],
                    MAX_TOKENS=gla_args["MAX_TOKENS"],
                    temp=gla_args["temp"]
                )
                        
            # Store result
            if subject_answer != "":
                self.results[question["id"]] = {
                    "question": question,
                    "subject_answer": subject_answer,
                    "is_correct": None,
                    "probs": probs 
                }
            self.total_count += 1
            print(f"Completed question {self.total_count}/{len(self.questions)}")
            if (i) % log_interval == 0: self._save_data()
            
        # Summary
        summary = f"\nCapabilities Test Complete."
        self._log(summary)
        
        self._save_data()
                    
        # Return the path to the capabilities data file
        capabilities_file_path = f"{self.log_base_name}{self.log_suffix}.json"
        self._log(f"Capabilities measurement completed. Results saved to: {capabilities_file_path}")
        return True, capabilities_file_path

def main(model_dataset_dict, temp):
    for subject_name, datasets in model_dataset_dict.items():
        for DATASET_NAME in datasets:
            IS_HUMAN = False
            INCLUDE_QNUM = False
            INCLUDE_TOTAL = False
            resume_from = None#"capabilities_1p_test_logs/llama-3.3-70b-instruct_SimpleMC_500_1759847064_test_data.json"#
            RESAMPLE = False
            NESTED = "All" #values: None, "Self", "Other", "All"
            EXPLICIT_CONFIDENCE_TASK = True  # If True, ask two separate prompts: direct question + confidence question
            temp = temp
            seed = 42
            
            # Set question count: use all available for large datasets, or sample for smaller ones
            if IS_HUMAN:
                N_QUESTIONS = 5
            elif DATASET_NAME.startswith("GP"):
                N_QUESTIONS = 447
            else:
                N_QUESTIONS = 500  # Default sample size 
            # Load questions first to get actual count
            formatted_questions = load_and_format_dataset(DATASET_NAME, N_QUESTIONS)
            if formatted_questions:
                actual_count = len(formatted_questions)
                SUBJECT_ID = f"{subject_name.replace('/', '-')}_{DATASET_NAME}_{actual_count}"
            else:
                print(f"Error: Failed to load dataset {DATASET_NAME}")
                continue
            
            try:
                # Questions already loaded above
                print(f"Using {actual_count} questions for capabilities measurement...")

                random.seed(seed)
                random.shuffle(formatted_questions)
                    
                # Use actual count if N_QUESTIONS was None, otherwise use the requested count
                questions_to_use = actual_count if N_QUESTIONS is None else min(N_QUESTIONS, actual_count)
                
                if not formatted_questions or len(formatted_questions) < questions_to_use:
                    print(f"Error: Not enough questions available ({len(formatted_questions) if formatted_questions else 0}). Needed: {questions_to_use}")
                    return
                
                # Create game instance for capabilities measurement
                game = CapabilitiesTest(
                    subject_id=SUBJECT_ID,
                    subject_name=subject_name,
                    questions=formatted_questions,
                    n_questions=questions_to_use,
                    is_human_player=IS_HUMAN,
                    resume_from=resume_from,
                    temperature=temp,
                    resample_for_probs=RESAMPLE,
                    nested=NESTED,
                    include_question_num=INCLUDE_QNUM,
                    include_total_questions=INCLUDE_TOTAL,
                    explicit_confidence_task=EXPLICIT_CONFIDENCE_TASK
                )

                # Store the seed used (run-level, for reproducibility)
                game.run_parameters["seed"] = seed
                
                # Set log suffix based on nested parameter and explicit_confidence_task
                if NESTED == "All":
                    game.log_suffix = "_explicit_confidence_task_all"
                elif EXPLICIT_CONFIDENCE_TASK:
                    game.log_suffix = "_explicit_confidence_task"
                elif NESTED == "Self":
                    game.log_suffix = "_explicit_confidence_task"
                elif NESTED == "Other":
                    game.log_suffix = "_explicit_confidence_task"  # or another suffix if different
                else:
                    game.log_suffix = "_test_data"
                            
                # Run capabilities measurement
                if (DATASET_NAME == "SimpleQA" or DATASET_NAME == "GPSA") and not NESTED:
                    success, capabilities_file = game.run_capabilities_measurement_sa()
                else:
                    success, capabilities_file = game.run_capabilities_measurement()
                
                if success:
                    print(f"\nCapabilities measurement completed successfully.")
                    print(f"Results saved to: {capabilities_file}")
                else:
                    print("\nCapabilities measurement failed.")
                    
            except Exception as e:
                print(f"Error during execution: {e}")
                import traceback
                traceback.print_exc()
    
    print("\nExecution completed.")

if __name__ == "__main__":
    model_dataset_dict = {
        # "llama-3.3-70b-instruct": ["PopMC_0_difficulty_filtered"], # Don't forget PopMC_0_difficulty_filtered vs PopMC
        "llama-3.1-405b-instruct": ["PopMC_0_difficulty_filtered"], # Don't forget PopMC_0_difficulty_filtered vs PopMC

        }
    main(model_dataset_dict, temp=1.0)