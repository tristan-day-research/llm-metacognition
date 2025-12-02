#!/usr/bin/env python3
"""
Verify that correct_answer_text in fine-tune logs matches correct_answer 
in validation data for each qid.
"""

import json
from pathlib import Path
from collections import defaultdict


def load_validation_data(val_file_path):
    """Load validation data and create qid -> correct_answer mapping."""
    qid_to_answer = {}
    with open(val_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                qid = data.get('qid')
                correct_answer = data.get('correct_answer')
                if qid and correct_answer:
                    qid_to_answer[qid] = correct_answer
    return qid_to_answer


def load_finetune_logs(logs_file_path):
    """Load fine-tune logs and extract mcq_accuracy_assessment entries."""
    entries = []
    with open(logs_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get('type') == 'mcq_accuracy_assessment':
                    entries.append(data)
    return entries


def verify_answer_matches(val_file_path, logs_file_path):
    """Verify that correct_answer_text matches correct_answer for each qid."""
    print(f"Loading validation data from: {val_file_path}")
    qid_to_answer = load_validation_data(val_file_path)
    print(f"Loaded {len(qid_to_answer)} questions from validation data")
    
    print(f"\nLoading fine-tune logs from: {logs_file_path}")
    log_entries = load_finetune_logs(logs_file_path)
    print(f"Loaded {len(log_entries)} mcq_accuracy_assessment entries")
    
    # Check validation steps
    validation_steps = set()
    for entry in log_entries:
        step = entry.get('validation_step')
        if step is not None:
            validation_steps.add(step)
    
    if validation_steps:
        print(f"Found {len(validation_steps)} validation steps: {sorted(validation_steps)[:10]}{'...' if len(validation_steps) > 10 else ''}")
        print(f"  (Each qid appears multiple times - once per validation step)")
    
    # Track matches and mismatches
    matches = []
    mismatches = []
    missing_qids = []
    
    # Group log entries by qid (in case there are duplicates)
    qid_to_log_entries = defaultdict(list)
    for entry in log_entries:
        qid = entry.get('qid')
        if qid:
            qid_to_log_entries[qid].append(entry)
    
    print(f"\nFound {len(qid_to_log_entries)} unique qids in fine-tune logs")
    
    # Calculate statistics about qid repetition
    qid_entry_counts = [len(entries) for entries in qid_to_log_entries.values()]
    if qid_entry_counts:
        print(f"  Entries per qid: min={min(qid_entry_counts)}, max={max(qid_entry_counts)}, avg={sum(qid_entry_counts)/len(qid_entry_counts):.2f}")
    
    # Track unique qids that match/mismatch
    unique_matching_qids = set()
    unique_mismatching_qids = set()
    
    # Check each qid
    for qid, log_entries_list in qid_to_log_entries.items():
        if qid not in qid_to_answer:
            missing_qids.append(qid)
            continue
        
        expected_answer = qid_to_answer[qid]
        qid_has_mismatch = False
        
        # Check all entries for this qid
        for entry in log_entries_list:
            correct_answer_text = entry.get('correct_answer_text')
            
            if correct_answer_text == expected_answer:
                matches.append((qid, expected_answer, correct_answer_text))
            else:
                qid_has_mismatch = True
                mismatches.append({
                    'qid': qid,
                    'expected': expected_answer,
                    'found': correct_answer_text,
                    'question': entry.get('question', 'N/A'),
                    'validation_step': entry.get('validation_step', 'N/A')
                })
        
        # Track unique qid status
        if qid_has_mismatch:
            unique_mismatching_qids.add(qid)
        else:
            unique_matching_qids.add(qid)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"Total log entries checked: {len(matches) + len(mismatches)}")
    print(f"  - Matching entries: {len(matches)}")
    print(f"  - Mismatching entries: {len(mismatches)}")
    print(f"\nUnique qids:")
    print(f"  - QIDs with all entries matching: {len(unique_matching_qids)}")
    print(f"  - QIDs with at least one mismatch: {len(unique_mismatching_qids)}")
    print(f"  - QIDs in logs but not in validation data: {len(missing_qids)}")
    print(f"\nNote: Each qid appears multiple times (once per validation step),")
    print(f"      so total entries > unique qids.")
    
    if missing_qids:
        print(f"\n{'='*80}")
        print(f"Missing QIDs (in logs but not in validation data):")
        print(f"{'='*80}")
        for qid in sorted(missing_qids)[:20]:  # Show first 20
            print(f"  - {qid}")
        if len(missing_qids) > 20:
            print(f"  ... and {len(missing_qids) - 20} more")
    
    if mismatches:
        print(f"\n{'='*80}")
        print(f"MISMATCHES (first 20):")
        print(f"{'='*80}")
        for i, mismatch in enumerate(mismatches[:20], 1):
            print(f"\n{i}. QID: {mismatch['qid']}")
            print(f"   Validation Step: {mismatch.get('validation_step', 'N/A')}")
            print(f"   Question: {mismatch['question']}")
            print(f"   Expected: {mismatch['expected']}")
            print(f"   Found:    {mismatch['found']}")
        
        if len(mismatches) > 20:
            print(f"\n... and {len(mismatches) - 20} more mismatches")
        
        # Save all mismatches to a file
        mismatches_file = Path(logs_file_path).parent / 'answer_mismatches.json'
        with open(mismatches_file, 'w', encoding='utf-8') as f:
            json.dump(mismatches, f, indent=2, ensure_ascii=False)
        print(f"\nAll {len(mismatches)} mismatches saved to: {mismatches_file}")
    else:
        print(f"\n✓ All answers match!")
    
    return {
        'matches': len(matches),
        'mismatches': len(mismatches),
        'missing_qids': len(missing_qids),
        'mismatch_details': mismatches
    }


if __name__ == '__main__':
    val_file = Path(__file__).parent / 'data' / 'PopMC_0_difficulty_filtered_val.jsonl'
    logs_file = Path(__file__).parent / 'fine_tune_logs' / 'meta-llama-Meta-Llama-3-8B-Instruct_2025-11-29-00-26-39_mcq_accuracy_assessment.jsonl'
    
    if not val_file.exists():
        print(f"Error: Validation file not found: {val_file}")
        exit(1)
    
    if not logs_file.exists():
        print(f"Error: Fine-tune logs file not found: {logs_file}")
        exit(1)
    
    verify_answer_matches(val_file, logs_file)

