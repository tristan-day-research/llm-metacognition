"""
finetune_delegate_game.py

Runs Delegate Game on both base (instruct) model and fine-tuned model using local inference.

This script:
  ✓ Loads base model and fine-tuned (LoRA) model using load_finetuned_model()
  ✓ Runs Delegate Game on both models for comparison
  ✓ Uses local transformers inference (no API calls)
  ✓ Takes parameters similar to finetune_run_finetuned_evaluations.py

Usage:
    python finetune_delegate_game.py \
        --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --lora_repo Tristan-Day/llama_3.1_finetuned \
        --capabilities_file compiled_results_smc/llama-3.1-8b-instruct_phase1_compiled.json \
        --dataset SimpleMC \
        --merge
"""

import json
import os
import torch
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
import string

from delegate_game_from_capabilities import DelegateGameFromCapabilities
from finetune_load_finetuned_model import load_finetuned_model, load_base_model
from finetune_prompting import build_multiple_choice_question_prompts, get_letter_token_ids
from finetune_utils import prepare_model_and_tokenizer, parse_letter_from_model_text, write_log


class LocalModelDelegateGame(DelegateGameFromCapabilities):
    """
    Delegate Game that uses local transformers models instead of API calls.
    Overrides _get_llm_answer to use local inference.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        **kwargs
    ):
        """
        Initialize with local model and tokenizer.
        
        Args:
            model: Loaded transformers model
            tokenizer: Loaded tokenizer
            device: Device to run inference on
            **kwargs: All other arguments passed to DelegateGameFromCapabilities
        """
        # Set a dummy subject_name for logging (will be overridden)
        kwargs.setdefault('subject_name', 'local-model')
        super().__init__(**kwargs)
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Track first question in each phase for debugging
        self._first_phase1_context_printed = False
        self._first_phase2_question_printed = False
        self._first_response_printed = False
        
        # Ensure model is on correct device and in eval mode
        self.model.eval()
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
    
    def _setup_provider(self):
        """Override to skip API provider setup since we use local models."""
        # Don't set up any API clients - we use local inference
        self.provider = "Local"
        self.client = None
        self._log("Using local model inference (no API provider needed)")
    
    def run_delegate_game(self):
        """Override to add Phase 1 context debugging."""
        # Print Phase 1 context before running
        if not self._first_phase1_context_printed:
            print("\n" + "="*80)
            print(" PHASE 1 CONTEXT - DEBUG OUTPUT")
            print("="*80)
            
            print("\n--- Initial Setup Explanation ---")
            print(self.initial_setup_explanation)
            
            if self.use_phase1_history:
                print("\n--- Phase 1 will use FULL HISTORY ---")
                print(f"Will create simulated history with {self.n_trials_phase1} questions")
                print(f"Subject accuracy in Phase 1: {self.subject_accuracy_phase1:.1%}")
                print(f"Teammate accuracy in Phase 1: {self.teammate_accuracy_phase1:.1%}")
                print("\nNote: Phase 1 history will be created during game execution.")
                print("First question format will be shown when Phase 2 starts.")
            elif self.use_phase1_summary:
                print("\n--- Phase 1 will use SUMMARY ---")
                summary_text = "\n\n" + "="*10 + " Phase 1 Summary " + "="*10
                if self.feedback_config['phase1_subject_feedback']:
                    summary_text += f"\nIn Phase 1, you answered {self.subject_accuracy_phase1:.1%} of the questions correctly."
                    summary_text += f"\nYour teammate answered {self.teammate_accuracy_phase1:.1%} of the questions correctly."
                else:
                    summary_text += f"\nIn Phase 1, your teammate answered {self.teammate_accuracy_phase1:.1%} of the questions correctly."
                summary_text += "\n" + "="*40 + "\n"
                print(f"Summary text that will be shown:\n{summary_text}")
            else:
                print("\n--- Phase 1: No context provided ---")
                print("Only initial setup explanation will be used for Phase 2")
            
            print("="*80 + "\n")
            self._first_phase1_context_printed = True
        
        # Call parent's run_delegate_game
        result = super().run_delegate_game()
        
        # After Phase 1 history is created, print first question if history was used
        if self.use_phase1_history and hasattr(self, 'phase1_simulated_history') and len(self.phase1_simulated_history) > 0:
            print("\n" + "="*80)
            print(" FIRST PHASE 1 QUESTION (SIMULATED) - DEBUG OUTPUT")
            print("="*80)
            first_user_msg = self.phase1_simulated_history[0]
            first_assistant_msg = self.phase1_simulated_history[1] if len(self.phase1_simulated_history) > 1 else None
            print("\n--- Simulated User Message (First Phase 1 Question) ---")
            print(first_user_msg.get('content', ''))
            if first_assistant_msg:
                print("\n--- Simulated Assistant Response ---")
                print(first_assistant_msg.get('content', ''))
            print("="*80 + "\n")
        
        return result
    
    def _record_trial(self, **kwargs):
        """Override to add progress reports every 10 Phase 2 questions."""
        # Call parent's _record_trial
        result = super()._record_trial(**kwargs)
        
        # Print progress every 10 Phase 2 questions
        if kwargs.get('phase') == 2:
            trial_num = kwargs.get('trial_num', 0)
            total_q = len(self.phase2_questions)
            
            if trial_num > 0 and (trial_num % 10) == 0:
                # Count correct answers so far
                correct_count = sum(1 for r in self.results 
                                  if r.get('phase') == 2 and 
                                  (r.get('subject_correct') is True or r.get('team_correct') is True))
                delegations = sum(1 for r in self.results 
                               if r.get('phase') == 2 and r.get('delegation_choice') == 'Teammate')
                self_answers_count = sum(1 for r in self.results 
                                       if r.get('phase') == 2 and r.get('delegation_choice') == 'Self')
                current_accuracy = (correct_count / trial_num) if trial_num > 0 else 0.0
                print(f"📊 Progress: Question {trial_num}/{total_q} | Score: {correct_count}/{trial_num} ({current_accuracy:.1%}) | Self: {self_answers_count} | Delegations: {delegations}")
        
        return result
    
    def _save_game_data(self, message_history=None):
        """Override to skip saving - we use our own logging system."""
        # Do nothing - we log to finetuned_evals/ instead
        pass
    
    def _get_llm_answer(
        self, 
        options, 
        q_text, 
        message_history, 
        keep_appending=True, 
        setup_text="", 
        MAX_TOKENS=1, 
        temp=0.0, 
        accept_any=True, 
        top_p=None, 
        top_k=None
    ):
        """
        Override to use local model inference instead of API calls.
        
        Args:
            options: List of valid option tokens (e.g., ["A", "B", "C", "D"] or ["1", "2"])
            q_text: Question text
            message_history: Previous conversation history
            keep_appending: Whether to append to history (not used for local models)
            setup_text: System prompt/instructions
            MAX_TOKENS: Max tokens to generate
            temp: Temperature for sampling
            accept_any: Whether to accept any response (for short answer)
            top_p: Top-p sampling (optional)
            top_k: Top-k sampling (optional)
        
        Returns:
            Tuple of (response_text, message_history, token_probs_dict)
        """
        # Build messages from history if provided
        if message_history and len(message_history) > 0:
            messages = message_history.copy()
            # Add current question
            if setup_text:
                messages.append({"role": "user", "content": setup_text + "\n\n" + q_text})
            else:
                messages.append({"role": "user", "content": q_text})
        else:
            # No history, start fresh
            if setup_text:
                full_prompt = setup_text + "\n\n" + q_text
            else:
                full_prompt = q_text
            messages = [{"role": "user", "content": full_prompt}]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Debug: Print first question in Phase 2
        # Check if this is Phase 2 by looking at the prompt content
        # Phase 2 prompts contain "Phase 2" or "Answer or Delegate" in the setup text or q_text
        full_text = (setup_text + "\n" + q_text).lower()
        is_phase2 = ("phase 2" in full_text or "answer or delegate" in full_text or 
                    ("delegate" in full_text and "choices:" in full_text))
        
        if is_phase2 and not self._first_phase2_question_printed:
            print("\n" + "="*80)
            print(" FIRST PHASE 2 QUESTION - DEBUG OUTPUT")
            print("="*80)
            print("\n--- EXACT PROMPT SENT TO MODEL ---")
            print(prompt)
            print("\n--- END PROMPT ---\n")
            self._first_phase2_question_printed = True
        
        # Tokenize
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            if options and not accept_any:
                # Multiple choice or decision-only: use logits to get probabilities
                out = self.model(**enc, use_cache=False)
                final_logits = out.logits[:, -1, :]  # [1, vocab_size]
                
                # Get token IDs for each option
                option_token_ids = {}
                for opt in options:
                    try:
                        # Try get_letter_token_ids first (works for A-D, T)
                        token_ids = get_letter_token_ids(self.tokenizer, opt)
                        option_token_ids[opt] = token_ids
                    except (ValueError, KeyError):
                        # For digits (1, 2) or other tokens, encode directly
                        ids = self.tokenizer.encode(opt, add_special_tokens=False)
                        if ids:
                            # Check if it's a single token
                            decoded = self.tokenizer.decode([ids[0]], skip_special_tokens=True)
                            if decoded.strip() == opt:
                                option_token_ids[opt] = [ids[0]]
                            else:
                                # Multi-token, take first token
                                option_token_ids[opt] = ids[:1]
                        else:
                            # Fallback: try with space
                            ids = self.tokenizer.encode(" " + opt, add_special_tokens=False)
                            if ids:
                                option_token_ids[opt] = ids[:1]
                
                # Aggregate logits for each option
                option_logits = {}
                for opt, token_ids in option_token_ids.items():
                    if token_ids:
                        letter_logits = final_logits[:, token_ids]
                        aggregated_logit = torch.logsumexp(letter_logits, dim=-1).item()
                        option_logits[opt] = aggregated_logit
                
                # Compute probabilities
                if option_logits:
                    logits_tensor = torch.tensor([option_logits[opt] for opt in options])
                    probs_tensor = torch.softmax(logits_tensor, dim=0)
                    # Always include all options in probabilities dict, even if 0.0
                    token_probs = {opt: probs_tensor[i].item() for i, opt in enumerate(options)}
                    
                    # Get predicted answer (argmax)
                    pred_idx = probs_tensor.argmax().item()
                    resp = options[pred_idx]
                else:
                    # Fallback: generate and parse
                    max_new_tokens = MAX_TOKENS if MAX_TOKENS else 10
                    generated = self.model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        temperature=temp if temp > 0 else None,
                        do_sample=temp > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    resp_text = self.tokenizer.decode(generated[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
                    resp = parse_letter_from_model_text(resp_text, options)
                    token_probs = {opt: 0.0 for opt in options}  # Unknown probabilities
            else:
                # Short answer or accept_any: generate text
                max_new_tokens = MAX_TOKENS if MAX_TOKENS else 50
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    temperature=temp if temp > 0 else None,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                resp = self.tokenizer.decode(generated[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                token_probs = {resp: 1.0} if resp else {}
        
        # Debug: Print first response in Phase 2
        full_text = (setup_text + "\n" + q_text).lower()
        is_phase2 = ("phase 2" in full_text or "answer or delegate" in full_text or 
                    ("delegate" in full_text and "choices:" in full_text))
        if is_phase2 and self._first_phase2_question_printed and not self._first_response_printed:
            print("\n--- EXACT MODEL RESPONSE ---")
            print(f"Response: {repr(resp)}")
            print(f"Token probabilities: {token_probs}")
            if options:
                print(f"Valid options: {options}")
                if resp in options:
                    print(f"✓ Response '{resp}' is a valid option")
                else:
                    print(f"⚠ Response '{resp}' is NOT in valid options {options}")
            print("--- END RESPONSE ---\n")
            print("="*80 + "\n")
            self._first_response_printed = True
        
        # Update message_history if keep_appending is True (matching parent class behavior)
        if keep_appending:
            message_history.append({"role": "assistant", "content": resp})
        
        return resp, message_history, token_probs


def run_delegate_game_with_local_model(
    base_model: str,
    lora_repo: Optional[str] = None,
    capabilities_file: str = None,
    dataset: str = "SimpleMC",
    merge: bool = False,
    evaluate_base_first: bool = True,
    # Delegate Game parameters
    n_trials_phase1: int = 50,
    n_trials_phase2: int = 500,
    teammate_accuracy_phase1: float = 0.5,
    teammate_accuracy_phase2: float = 0.5,
    decision_only: bool = True,
    use_phase1_summary: bool = True,
    use_phase1_history: bool = False,
    temperature: float = 0.0,
    seed: int = 33,
    # Other parameters
    device: str = "cuda",
):
    """
    Run Delegate Game on base and/or fine-tuned models using local inference.
    
    Args:
        base_model: Base model name (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
        lora_repo: HuggingFace repo or local path containing LoRA weights (optional)
        capabilities_file: Path to Phase 1 completed results file
        dataset: Dataset name (e.g., "SimpleMC", "GPQA")
        merge: If True, merge LoRA weights into base model
        evaluate_base_first: If True, run base model first, then fine-tuned
        n_trials_phase1: Number of Phase 1 questions
        n_trials_phase2: Number of Phase 2 questions
        teammate_accuracy_phase1: Teammate accuracy in Phase 1
        teammate_accuracy_phase2: Teammate accuracy in Phase 2
        decision_only: If True, decision-only mode (Answer/Delegate choice only)
        use_phase1_summary: If True, provide Phase 1 summary
        use_phase1_history: If True, provide full Phase 1 history
        temperature: Temperature for generation
        seed: Random seed
        device: Device to run on ("cuda" or "cpu")
        
    Note: All results are logged to finetuned_evals/ directory (same as other finetune scripts)
    
    Returns:
        Dict with results for base and/or fine-tuned models
    """
    print("🚀 Starting Delegate Game evaluation...")
    print(f"   Base model: {base_model}")
    if lora_repo:
        print(f"   LoRA repo: {lora_repo}")
    print(f"   Dataset: {dataset}")
    print(f"   Device: {device}\n")
    
    results = {}
    
    # Extract dataset name from path if it's a file path, otherwise use as-is
    dataset_name = dataset
    if os.path.exists(dataset) or dataset.endswith('.jsonl') or dataset.endswith('.json'):
        # It's a file path - extract name for logging
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        # Remove common suffixes
        dataset_name = dataset_name.replace('_test', '').replace('_train', '').replace('_val', '')
        print(f"📂 Dataset path provided: {dataset}")
        print(f"   Using dataset name for logging: {dataset_name}")
    
    # Determine capabilities file path if not provided
    if capabilities_file is None:
        # Try to infer from base_model name
        model_name_safe = base_model.split("/")[-1].replace("_", "-")
        if dataset_name == "SimpleMC":
            capabilities_file = f"./compiled_results_smc/{model_name_safe}_phase1_compiled.json"
        elif dataset_name == "SimpleQA":
            capabilities_file = f"./compiled_results_sqa/{model_name_safe}_phase1_compiled.json"
        else:
            capabilities_file = f"./completed_results_{dataset_name.lower()}/{model_name_safe}_phase1_completed.json"
    
    print(f"📂 Looking for capabilities file: {capabilities_file}")
    if not os.path.exists(capabilities_file):
        raise FileNotFoundError(
            f"Capabilities file not found: {capabilities_file}\n"
            f"Please run capabilities test first or provide --capabilities_file"
        )
    print(f"✓ Found capabilities file\n")
    
    # Setup logging directory (use finetuned_evals like other finetune scripts)
    log_dir = "finetuned_evals"  # Override to use standard directory
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    model_name_safe = base_model.split("/")[-1].replace("_", "-")
    lora_name = lora_repo.split("/")[-1] if lora_repo else "base-only"
    
    # Setup log file paths (similar to finetune_run_finetuned_evaluations.py)
    if evaluate_base_first and lora_repo:
        # Use a single log file for both evaluations
        combined_log_file = os.path.join(
            log_dir,
            f"delegate_game_{timestamp}_{model_name_safe}_{lora_name}_{dataset_name}_n{n_trials_phase2}_comparison.jsonl"
        )
        print(f"📝 Logging both models to: {combined_log_file}")
    elif lora_repo:
        # Fine-tuned only
        combined_log_file = os.path.join(
            log_dir,
            f"delegate_game_{timestamp}_{model_name_safe}_{lora_name}_{dataset_name}_n{n_trials_phase2}.jsonl"
        )
        print(f"📝 Logging fine-tuned model results to: {combined_log_file}")
    else:
        # Base only
        combined_log_file = os.path.join(
            log_dir,
            f"delegate_game_{timestamp}_{model_name_safe}_{dataset_name}_n{n_trials_phase2}_base.jsonl"
        )
        print(f"📝 Logging base model results to: {combined_log_file}")
    
    # Log all parameters as the first entry
    parameters_entry = {
        "type": "parameters",
        "base_model": base_model,
        "lora_repo": lora_repo,
        "capabilities_file": capabilities_file,
        "dataset": dataset,
        "dataset_name": dataset_name,
        "merge": merge,
        "evaluate_base_first": evaluate_base_first,
        "n_trials_phase1": n_trials_phase1,
        "n_trials_phase2": n_trials_phase2,
        "teammate_accuracy_phase1": teammate_accuracy_phase1,
        "teammate_accuracy_phase2": teammate_accuracy_phase2,
        "decision_only": decision_only,
        "use_phase1_summary": use_phase1_summary,
        "use_phase1_history": use_phase1_history,
        "temperature": temperature,
        "seed": seed,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_log(combined_log_file, parameters_entry)
    
    # Build subject ID suffix
    suffix = ""
    if not use_phase1_history:
        suffix += "_nohistory"
    if use_phase1_summary:
        suffix += "_summary"
    if decision_only:
        suffix += "_decisionOnly"
    suffix += f"_team{teammate_accuracy_phase2}_temp{temperature}"
    
    # Feedback config
    feedback_config = {
        'phase1_subject_feedback': False,
        'phase1_teammate_feedback': True,
        'phase2_subject_feedback': False,
        'phase2_teammate_feedback': False,
        'show_answer_with_correctness': False,
    }
    
    # Setup explanation
    qtypestr = "multiple-choice" if dataset_name not in ["SimpleQA", "GPSA"] else "short-answer"
    if decision_only:
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
        if use_phase1_history:
            initial_setup_explanation += "In phase 1, you will receive feedback on your and your teammate's performance. Use this feedback to inform your decisions about whether to delegate in phase 2."
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
        if use_phase1_history:
            initial_setup_explanation += "In phase 1, you will receive feedback on your and your teammate's performance. Use this feedback to inform your decisions about whether to delegate in phase 2."
        initial_setup_explanation += """
        You want to maximize your team's total score. In phase 1, that means answering as best you can. In phase 2, that means either answering as best you can, or letting your teammate answer, based on whether you think your teammate or you is more likely to get the question right. Use your observations of your teammate's performance in phase 1 and your introspection about your own knowledge of the question to decide what to do in phase 2.
        """
    
    # Optionally evaluate base model first
    if evaluate_base_first:
        print("\n" + "="*60)
        print(" RUNNING DELEGATE GAME ON BASE (UN-FINETUNED) MODEL")
        print("="*60 + "\n")
        
        print("📥 Loading base model...")
        base_model_obj, base_tokenizer = load_base_model(base_model)
        print("✓ Base model loaded\n")
        base_model_obj, base_tokenizer = prepare_model_and_tokenizer(base_model_obj, base_tokenizer)
        
        base_subject_id = f"{model_name_safe}_{dataset_name}_{n_trials_phase1}_{n_trials_phase2}_base{suffix}"
        # Sanitize subject_id to avoid path issues (remove slashes, etc.)
        base_subject_id = base_subject_id.replace("/", "-").replace("\\", "-")
        
        base_game = LocalModelDelegateGame(
            model=base_model_obj,
            tokenizer=base_tokenizer,
            device=device,
            subject_id=base_subject_id,
            subject_name=f"{base_model}_base",
            is_human_player=False,
            completed_results_file=capabilities_file,
            dataset=dataset_name,
            log_dir=None,  # Disable base game class logging (we use our own)
            n_trials_phase1=n_trials_phase1,
            n_trials_phase2=n_trials_phase2,
            teammate_accuracy_phase1=teammate_accuracy_phase1,
            teammate_accuracy_phase2=teammate_accuracy_phase2,
            feedback_config=feedback_config,
            use_phase1_summary=use_phase1_summary,
            use_phase1_history=use_phase1_history,
            initial_setup_explanation=initial_setup_explanation,
            seed=seed,
            temperature=temperature,
            decision_only=decision_only,
            alternate_decision_mapping=False,
        )
        
        base_results = base_game.run_delegate_game()
        
        # Log all Phase 2 trials
        log_prefix = "base_"
        for trial in base_results:
            if trial.get("phase") == 2:  # Only log Phase 2 trials
                log_entry = {
                    "type": f"{log_prefix}delegate_game_trial",
                    "trial_in_phase": trial.get("trial_in_phase"),
                    "question_id": trial.get("question_id"),
                    "question_text": trial.get("question_text"),
                    "correct_answer": trial.get("correct_answer"),
                    "delegation_choice": trial.get("delegation_choice"),
                    "subject_answer": trial.get("subject_answer"),
                    "subject_correct": trial.get("subject_correct"),
                    "teammate_answer": trial.get("teammate_answer"),
                    "teammate_correct": trial.get("teammate_correct"),
                    "team_answer": trial.get("team_answer"),
                    "team_correct": trial.get("team_correct"),
                    "options": trial.get("options"),
                    "probs": trial.get("probs"),
                    "timestamp": trial.get("timestamp"),
                }
                write_log(combined_log_file, log_entry)
        
        # Log summary
        team_delegations = sum(1 for r in base_results if r.get('phase') == 2 and r.get('delegation_choice') == 'Teammate')
        self_answers = sum(1 for r in base_results if r.get('phase') == 2 and r.get('delegation_choice') == 'Self')
        self_correct = sum(1 for r in base_results if r.get('phase') == 2 and r.get('delegation_choice') == 'Self' and r.get('team_correct'))
        team_correct = sum(1 for r in base_results if r.get('phase') == 2 and r.get('delegation_choice') == 'Teammate' and r.get('team_correct'))
        
        summary_entry = {
            "type": f"{log_prefix}delegate_game_summary",
            "subject_id": base_game.subject_id,
            "dataset": dataset_name,
            "n_trials_phase1": n_trials_phase1,
            "n_trials_phase2": n_trials_phase2,
            "teammate_accuracy_phase1": teammate_accuracy_phase1,
            "teammate_accuracy_phase2": teammate_accuracy_phase2,
            "decision_only": decision_only,
            "use_phase1_summary": use_phase1_summary,
            "use_phase1_history": use_phase1_history,
            "temperature": temperature,
            "seed": seed,
            "phase2_accuracy": base_game.phase2_accuracy,
            "phase2_score": base_game.phase2_score,
            "total_phase2_trials": len([r for r in base_results if r.get("phase") == 2]),
            "delegations_to_teammate": team_delegations,
            "self_answers": self_answers,
            "self_answer_accuracy": (self_correct / self_answers) if self_answers > 0 else 0.0,
            "teammate_delegation_accuracy": (team_correct / team_delegations) if team_delegations > 0 else 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_log(combined_log_file, summary_entry)
        
        results['base'] = {
            'results': base_results,
            'phase2_accuracy': base_game.phase2_accuracy,
            'phase2_score': base_game.phase2_score,
            'game_data_file': getattr(base_game, 'game_data_filename', None),
            'log_file': combined_log_file,
        }
        
        print(f"\n✓ Base model game completed.")
        print(f"✓ Logged to: {combined_log_file}")
        
        # Clean up
        del base_model_obj
        del base_tokenizer
        del base_game
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\n" + "="*60)
        print(" BASE MODEL GAME COMPLETE")
        print("="*60 + "\n")
    
    # Run fine-tuned model if LoRA repo provided
    if lora_repo:
        print("\n" + "="*60)
        print(" RUNNING DELEGATE GAME ON FINETUNED MODEL")
        print("="*60 + "\n")
        
        print("📥 Loading fine-tuned model...")
        finetuned_model, finetuned_tokenizer = load_finetuned_model(
            base_model=base_model,
            lora_repo=lora_repo,
            merge=merge,
        )
        finetuned_model, finetuned_tokenizer = prepare_model_and_tokenizer(finetuned_model, finetuned_tokenizer)
        print("✓ Fine-tuned model loaded\n")
        
        finetuned_subject_id = f"{model_name_safe}_{lora_name}_{dataset_name}_{n_trials_phase1}_{n_trials_phase2}{suffix}"
        # Sanitize subject_id to avoid path issues (remove slashes, etc.)
        finetuned_subject_id = finetuned_subject_id.replace("/", "-").replace("\\", "-")
        
        finetuned_game = LocalModelDelegateGame(
            model=finetuned_model,
            tokenizer=finetuned_tokenizer,
            device=device,
            subject_id=finetuned_subject_id,
            subject_name=f"{base_model}_{lora_name}",
            is_human_player=False,
            completed_results_file=capabilities_file,
            dataset=dataset_name,
            log_dir=None,  # Disable base game class logging (we use our own)
            n_trials_phase1=n_trials_phase1,
            n_trials_phase2=n_trials_phase2,
            teammate_accuracy_phase1=teammate_accuracy_phase1,
            teammate_accuracy_phase2=teammate_accuracy_phase2,
            feedback_config=feedback_config,
            use_phase1_summary=use_phase1_summary,
            use_phase1_history=use_phase1_history,
            initial_setup_explanation=initial_setup_explanation,
            seed=seed,
            temperature=temperature,
            decision_only=decision_only,
            alternate_decision_mapping=False,
        )
        
        finetuned_results = finetuned_game.run_delegate_game()
        
        # Log all Phase 2 trials
        log_prefix = "finetuned_"
        for trial in finetuned_results:
            if trial.get("phase") == 2:  # Only log Phase 2 trials
                log_entry = {
                    "type": f"{log_prefix}delegate_game_trial",
                    "trial_in_phase": trial.get("trial_in_phase"),
                    "question_id": trial.get("question_id"),
                    "question_text": trial.get("question_text"),
                    "correct_answer": trial.get("correct_answer"),
                    "delegation_choice": trial.get("delegation_choice"),
                    "subject_answer": trial.get("subject_answer"),
                    "subject_correct": trial.get("subject_correct"),
                    "teammate_answer": trial.get("teammate_answer"),
                    "teammate_correct": trial.get("teammate_correct"),
                    "team_answer": trial.get("team_answer"),
                    "team_correct": trial.get("team_correct"),
                    "options": trial.get("options"),
                    "probs": trial.get("probs"),
                    "timestamp": trial.get("timestamp"),
                }
                write_log(combined_log_file, log_entry)
        
        # Log summary
        team_delegations = sum(1 for r in finetuned_results if r.get('phase') == 2 and r.get('delegation_choice') == 'Teammate')
        self_answers = sum(1 for r in finetuned_results if r.get('phase') == 2 and r.get('delegation_choice') == 'Self')
        self_correct = sum(1 for r in finetuned_results if r.get('phase') == 2 and r.get('delegation_choice') == 'Self' and r.get('team_correct'))
        team_correct = sum(1 for r in finetuned_results if r.get('phase') == 2 and r.get('delegation_choice') == 'Teammate' and r.get('team_correct'))
        
        summary_entry = {
            "type": f"{log_prefix}delegate_game_summary",
            "subject_id": finetuned_game.subject_id,
            "dataset": dataset_name,
            "n_trials_phase1": n_trials_phase1,
            "n_trials_phase2": n_trials_phase2,
            "teammate_accuracy_phase1": teammate_accuracy_phase1,
            "teammate_accuracy_phase2": teammate_accuracy_phase2,
            "decision_only": decision_only,
            "use_phase1_summary": use_phase1_summary,
            "use_phase1_history": use_phase1_history,
            "temperature": temperature,
            "seed": seed,
            "phase2_accuracy": finetuned_game.phase2_accuracy,
            "phase2_score": finetuned_game.phase2_score,
            "total_phase2_trials": len([r for r in finetuned_results if r.get("phase") == 2]),
            "delegations_to_teammate": team_delegations,
            "self_answers": self_answers,
            "self_answer_accuracy": (self_correct / self_answers) if self_answers > 0 else 0.0,
            "teammate_delegation_accuracy": (team_correct / team_delegations) if team_delegations > 0 else 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_log(combined_log_file, summary_entry)
        
        results['finetuned'] = {
            'results': finetuned_results,
            'phase2_accuracy': finetuned_game.phase2_accuracy,
            'phase2_score': finetuned_game.phase2_score,
            'game_data_file': getattr(finetuned_game, 'game_data_filename', None),
            'log_file': combined_log_file,
        }
        
        print(f"\n✓ Fine-tuned model game completed.")
        print(f"✓ Logged to: {combined_log_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    def str_to_bool(v):
        """Convert string to boolean."""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')
    
    parser = argparse.ArgumentParser(description="Run Delegate Game on fine-tuned Llama model")
    parser.add_argument("--base_model", required=True, 
                       help="Base model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--lora_repo", type=str, default=None,
                       help="HuggingFace repo or local path containing LoRA weights")
    parser.add_argument("--capabilities_file", type=str, default=None,
                       help="Path to Phase 1 completed results file (auto-detected if not provided)")
    parser.add_argument("--dataset", type=str, default="SimpleMC",
                       help="Dataset name (e.g., SimpleMC, GPQA) or path to dataset file (e.g., data/PopMC.jsonl)")
    parser.add_argument("--merge", action="store_true",
                       help="Merge LoRA weights into base model")
    parser.add_argument("--no_base", action="store_true",
                       help="Skip base model evaluation (only run fine-tuned)")
    
    # Delegate Game parameters (all required)
    parser.add_argument("--n_trials_phase1", type=int, required=True,
                       help="Number of Phase 1 questions")
    parser.add_argument("--n_trials_phase2", type=int, required=True,
                       help="Number of Phase 2 questions")
    parser.add_argument("--teammate_accuracy_phase1", type=float, required=True,
                       help="Teammate accuracy in Phase 1 (0.0 to 1.0)")
    parser.add_argument("--teammate_accuracy_phase2", type=float, required=True,
                       help="Teammate accuracy in Phase 2 (0.0 to 1.0)")
    parser.add_argument("--decision_only", type=str_to_bool, required=True,
                       help="Decision-only mode: True/true/yes/1 for Answer/Delegate choice only, False/false/no/0 for full answer mode")
    parser.add_argument("--phase1_summary", type=str_to_bool, required=True,
                       help="Provide Phase 1 summary: True/true/yes/1 to show summary, False/false/no/0 to skip")
    parser.add_argument("--use_phase1_history", type=str_to_bool, required=True,
                       help="Provide full Phase 1 history: True/true/yes/1 for full history, False/false/no/0 for summary only")
    parser.add_argument("--temperature", type=float, required=True,
                       help="Temperature for generation (0.0 for deterministic)")
    parser.add_argument("--seed", type=int, required=True,
                       help="Random seed for reproducibility")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to run on")
    
    args = parser.parse_args()
    
    print("="*60)
    print(" DELEGATE GAME - FINE-TUNED MODEL EVALUATION")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"LoRA repo: {args.lora_repo}")
    print(f"Dataset: {args.dataset}")
    print(f"Phase 1 trials: {args.n_trials_phase1}")
    print(f"Phase 2 trials: {args.n_trials_phase2}")
    print(f"Decision only: {args.decision_only}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")
    
    if not args.lora_repo and args.no_base:
        raise ValueError("Must provide --lora_repo if --no_base is set")
    
    try:
        results = run_delegate_game_with_local_model(
        base_model=args.base_model,
        lora_repo=args.lora_repo,
        capabilities_file=args.capabilities_file,
        dataset=args.dataset,
        merge=args.merge,
        evaluate_base_first=not args.no_base,
        n_trials_phase1=args.n_trials_phase1,
        n_trials_phase2=args.n_trials_phase2,
        teammate_accuracy_phase1=args.teammate_accuracy_phase1,
        teammate_accuracy_phase2=args.teammate_accuracy_phase2,
        decision_only=args.decision_only,
        use_phase1_summary=args.phase1_summary,
        use_phase1_history=args.use_phase1_history,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
        )
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "="*60)
    print(" FINAL RESULTS")
    print("="*60)
    
    if 'base' in results:
        base_acc = results['base']['phase2_accuracy']
        print(f"Base Model Phase 2 Accuracy: {base_acc:.4f} ({base_acc*100:.2f}%)")
        print(f"  Results file: {results['base']['game_data_file']}")
        print(f"  Log file: {results['base'].get('log_file', 'N/A')}")
    
    if 'finetuned' in results:
        finetuned_acc = results['finetuned']['phase2_accuracy']
        print(f"Fine-tuned Model Phase 2 Accuracy: {finetuned_acc:.4f} ({finetuned_acc*100:.2f}%)")
        print(f"  Results file: {results['finetuned']['game_data_file']}")
        print(f"  Log file: {results['finetuned'].get('log_file', 'N/A')}")
    
    if 'base' in results and 'finetuned' in results:
        improvement = finetuned_acc - base_acc
        print(f"\nImprovement: {improvement:+.4f} ({improvement*100:+.2f}%)")

