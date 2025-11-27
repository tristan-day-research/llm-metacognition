#!/usr/bin/env python3
"""
Script to count the number of questions in JSON or JSONL files
"""

import json
import sys
import os
from collections import Counter

def count_questions(file_path):
    """Count the number of questions in the JSON or JSONL file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return 0
    
    # Check if it's a JSONL file (by extension)
    is_jsonl = file_path.endswith('.jsonl')
    
    # Try JSONL format first if extension suggests it, or if JSON parsing fails
    if is_jsonl:
        return count_jsonl(file_path)
    else:
        # Try regular JSON first
        try:
            print(f"Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's a list
            if isinstance(data, list):
                count = len(data)
                print(f"\nTotal number of questions: {count}")
                
                # Check for duplicates in list format
                check_duplicates_list(data, file_path)
                
                return count
            elif isinstance(data, dict):
                # If it's a dict, check for common keys that might contain the questions
                if 'Data' in data:
                    questions = data['Data']
                    if isinstance(questions, list):
                        count = len(questions)
                        print(f"\nTotal number of questions: {count}")
                        return count
                    elif isinstance(questions, dict):
                        count = len(questions)
                        print(f"\nTotal number of questions: {count}")
                        return count
                    else:
                        print(f"Error: 'Data' key contains {type(questions)}, expected a list or dict")
                        return 0
                elif 'questions' in data:
                    questions = data['questions']
                    if isinstance(questions, list):
                        count = len(questions)
                        print(f"\nTotal number of questions: {count}")
                        return count
                    elif isinstance(questions, dict):
                        count = len(questions)
                        print(f"\nTotal number of questions: {count}")
                        return count
                    else:
                        print(f"Error: 'questions' key contains {type(questions)}, expected a list or dict")
                        return 0
                elif 'results' in data:
                    results = data['results']
                    if isinstance(results, list):
                        count = len(results)
                        print(f"\nTotal number of questions: {count}")
                        return count
                    elif isinstance(results, dict):
                        # Count the number of keys in the results dictionary
                        count = len(results)
                        print(f"\nTotal number of questions: {count}")
                        
                        # Check for duplicate question IDs
                        check_duplicates(results, file_path)
                        
                        return count
                    else:
                        print(f"Error: 'results' key contains {type(results)}, expected a list or dict")
                        return 0
                else:
                    # Fallback: if it's a dictionary without expected keys, count top-level keys
                    # (useful for files where each key is a response)
                    count = len(data)
                    print(f"\nTotal number of questions (counted as top-level keys): {count}")
                    print(f"Available keys: {list(data.keys())[:10]}{'...' if len(data.keys()) > 10 else ''}")
                    return count
            else:
                print(f"Error: Unexpected data type: {type(data)}")
                return 0
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try JSONL format
            print("JSON parsing failed, trying JSONL format...")
            return count_jsonl(file_path)
        except Exception as e:
            print(f"Error: {e}")
            return 0

def check_duplicates_list(data_list, file_path):
    """Check for duplicate question IDs and duplicate question text in a list of question objects."""
    question_ids = []
    question_texts = []
    id_to_indices = {}  # Map question ID to list of indices/positions
    text_to_indices = {}  # Map question text to list of indices/positions
    
    # Collect all question IDs and texts from the list
    for idx, item in enumerate(data_list):
        if isinstance(item, dict):
            # Check for question ID (try different possible keys)
            question_id = None
            if 'id' in item:
                question_id = item['id']
            elif 'qid' in item:
                question_id = item['qid']
            elif 'question_id' in item:
                question_id = item['question_id']
            
            if question_id is not None:
                question_ids.append(question_id)
                if question_id not in id_to_indices:
                    id_to_indices[question_id] = []
                id_to_indices[question_id].append(idx)
            
            # Check for question text
            question_text = None
            if 'question' in item:
                question_text = item['question']
            elif 'Question' in item:
                question_text = item['Question']
            
            if question_text:
                question_text = question_text.strip()
                question_texts.append(question_text)
                if question_text not in text_to_indices:
                    text_to_indices[question_text] = []
                text_to_indices[question_text].append(idx)
    
    # Check for duplicate IDs
    id_counts = Counter(question_ids)
    id_duplicates = {id: count for id, count in id_counts.items() if count > 1}
    total_duplicate_ids = sum(count - 1 for count in id_duplicates.values()) if id_duplicates else 0
    
    # Check for duplicate question texts
    text_counts = Counter(question_texts)
    text_duplicates = {text: count for text, count in text_counts.items() if count > 1}
    total_duplicate_questions = sum(count - 1 for count in text_duplicates.values()) if text_duplicates else 0
    
    # Always print duplicate counts
    print(f"\nDuplicate question IDs: {len(id_duplicates)} unique, {total_duplicate_ids} total")
    print(f"Duplicate question texts: {len(text_duplicates)} unique, {total_duplicate_questions} total")
    
    # for dup_id, count in sorted(id_duplicates.items()):
    #     print(f"  - Question ID '{dup_id}' appears {count} time(s)")
    #     if dup_id in id_to_indices:
    #         indices_list = id_to_indices[dup_id]
    #         print(f"    Found at indices: {', '.join(str(i) for i in indices_list[:10])}{'...' if len(indices_list) > 10 else ''}")
    # for dup_text, count in sorted(text_duplicates.items(), key=lambda x: x[1], reverse=True):
    #     # Truncate long question text for display
    #     display_text = dup_text[:80] + '...' if len(dup_text) > 80 else dup_text
    #     print(f"  - Question text '{display_text}' appears {count} time(s)")
    #     if dup_text in text_to_indices:
    #         indices_list = text_to_indices[dup_text]
    #         print(f"    Found at indices: {', '.join(str(i) for i in indices_list[:10])}{'...' if len(indices_list) > 10 else ''}")

def check_duplicates(results, file_path):
    """Check for duplicate question IDs and duplicate question text in the results dictionary."""
    question_ids = []
    question_texts = []
    id_to_keys = {}  # Map question ID to list of keys that contain it
    text_to_keys = {}  # Map question text to list of keys that contain it
    
    # Collect all question IDs and texts from the question objects
    for key, value in results.items():
        if isinstance(value, dict) and 'question' in value:
            question_obj = value['question']
            if isinstance(question_obj, dict):
                # Check for question ID
                if 'id' in question_obj:
                    question_id = question_obj['id']
                    question_ids.append(question_id)
                    
                    # Track which keys contain this question ID
                    if question_id not in id_to_keys:
                        id_to_keys[question_id] = []
                    id_to_keys[question_id].append(key)
                    
                    # Also check if the key matches the id
                    if question_id != key:
                        print(f"  Note: Key '{key}' has question.id '{question_id}' (mismatch)")
                
                # Check for question text
                if 'question' in question_obj:
                    question_text = question_obj['question'].strip()
                    question_texts.append(question_text)
                    
                    # Track which keys contain this question text
                    if question_text not in text_to_keys:
                        text_to_keys[question_text] = []
                    text_to_keys[question_text].append(key)
    
    # Check for duplicate IDs
    id_counts = Counter(question_ids)
    id_duplicates = {id: count for id, count in id_counts.items() if count > 1}
    total_duplicate_ids = sum(count - 1 for count in id_duplicates.values()) if id_duplicates else 0
    
    # Check for duplicate question texts
    text_counts = Counter(question_texts)
    text_duplicates = {text: count for text, count in text_counts.items() if count > 1}
    total_duplicate_questions = sum(count - 1 for count in text_duplicates.values()) if text_duplicates else 0
    
    # Always print duplicate counts
    print(f"\nDuplicate question IDs: {len(id_duplicates)} unique, {total_duplicate_ids} total")
    print(f"Duplicate question texts: {len(text_duplicates)} unique, {total_duplicate_questions} total")
    
    # for dup_id, count in sorted(id_duplicates.items()):
    #     print(f"  - Question ID '{dup_id}' appears {count} time(s)")
    #     if dup_id in id_to_keys:
    #         keys_list = id_to_keys[dup_id]
    #         print(f"    Found in keys: {', '.join(keys_list[:10])}{'...' if len(keys_list) > 10 else ''}")
    # for dup_text, count in sorted(text_duplicates.items(), key=lambda x: x[1], reverse=True):
    #     # Truncate long question text for display
    #     display_text = dup_text[:80] + '...' if len(dup_text) > 80 else dup_text
    #     print(f"  - Question text '{display_text}' appears {count} time(s)")
    #     if dup_text in text_to_keys:
    #         keys_list = text_to_keys[dup_text]
    #         print(f"    Found in keys: {', '.join(keys_list[:10])}{'...' if len(keys_list) > 10 else ''}")

def check_duplicates_jsonl(questions_data, file_path):
    """Check for duplicate questions in JSONL data (list of dicts)."""
    question_ids = []
    question_texts = []
    id_to_indices = {}  # Map question ID to list of indices/positions
    text_to_indices = {}  # Map question text to list of indices/ids
    
    # Collect all question IDs and texts
    for idx, item in enumerate(questions_data):
        if isinstance(item, dict):
            # Try different possible keys for question text
            question_text = None
            question_id = None
            
            if 'question' in item:
                question_text = item['question'].strip()
            elif 'Question' in item:
                question_text = item['Question'].strip()
            
            # Get question ID for tracking
            if 'qid' in item:
                question_id = item['qid']
            elif 'id' in item:
                question_id = item['id']  # Keep as original type (could be int or string)
            else:
                question_id = f"line_{idx + 1}"
            
            # Track question ID
            if question_id is not None:
                question_ids.append(question_id)
                if question_id not in id_to_indices:
                    id_to_indices[question_id] = []
                id_to_indices[question_id].append(idx)
            
            if question_text:
                question_texts.append(question_text)
                if question_text not in text_to_indices:
                    text_to_indices[question_text] = []
                text_to_indices[question_text].append(question_id if question_id else f"line_{idx + 1}")
    
    # Check for duplicate IDs
    id_counts = Counter(question_ids)
    id_duplicates = {id: count for id, count in id_counts.items() if count > 1}
    total_duplicate_ids = sum(count - 1 for count in id_duplicates.values()) if id_duplicates else 0
    
    # Check for duplicate question texts
    text_counts = Counter(question_texts)
    text_duplicates = {text: count for text, count in text_counts.items() if count > 1}
    total_duplicate_questions = sum(count - 1 for count in text_duplicates.values()) if text_duplicates else 0
    
    # Always print duplicate counts
    print(f"\nDuplicate question IDs: {len(id_duplicates)} unique, {total_duplicate_ids} total")
    print(f"Duplicate question texts: {len(text_duplicates)} unique, {total_duplicate_questions} total")
    
    # for dup_id, count in sorted(id_duplicates.items()):
    #     print(f"  - Question ID '{dup_id}' appears {count} time(s)")
    #     if dup_id in id_to_indices:
    #         indices_list = id_to_indices[dup_id]
    #         print(f"    Found at indices: {', '.join(str(i) for i in indices_list[:10])}{'...' if len(indices_list) > 10 else ''}")
    # for dup_text, count in sorted(text_duplicates.items(), key=lambda x: x[1], reverse=True):
    #     # Truncate long question text for display
    #     display_text = dup_text[:80] + '...' if len(dup_text) > 80 else dup_text
    #     print(f"  - Question text '{display_text}' appears {count} time(s)")
    #     if dup_text in text_to_indices:
    #         ids_list = text_to_indices[dup_text]
    #         print(f"    Found in IDs: {', '.join(str(id) for id in ids_list[:10])}{'...' if len(ids_list) > 10 else ''}")

def remove_duplicate_questions(file_path):
    """Remove duplicate questions from a JSON or JSONL file based on question text match.
    
    Outputs a new file with '_duplicate_questions_removed' suffix.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    
    # Determine output file name
    base_name, ext = os.path.splitext(file_path)
    output_path = f"{base_name}_duplicate_questions_removed{ext}"
    
    print(f"Loading {file_path}...")
    
    # Check if it's a JSONL file
    is_jsonl = file_path.endswith('.jsonl')
    
    if is_jsonl:
        # Handle JSONL format
        questions_data = []
        seen_question_texts = set()
        removed_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # Get question text
                        question_text = None
                        if 'question' in data:
                            question_text = data['question'].strip()
                        elif 'Question' in data:
                            question_text = data['Question'].strip()
                        
                        if question_text:
                            if question_text not in seen_question_texts:
                                seen_question_texts.add(question_text)
                                questions_data.append(data)
                            else:
                                removed_count += 1
                        else:
                            # If no question text found, keep it
                            questions_data.append(data)
                        
                        if line_num % 10000 == 0:
                            print(f"Processed {line_num} lines, kept {len(questions_data)} questions so far...")
                    except json.JSONDecodeError:
                        continue
        
        # Write output JSONL file
        print(f"Writing output to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in questions_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Removed {removed_count} duplicate questions.")
        print(f"Kept {len(questions_data)} unique questions.")
        print(f"Output saved to: {output_path}")
        
    else:
        # Handle JSON format
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                if 'results' in data:
                    # Structure with results dictionary
                    results = data['results']
                    if isinstance(results, dict):
                        seen_question_texts = set()
                        filtered_results = {}
                        removed_count = 0
                        
                        for key, value in results.items():
                            if isinstance(value, dict) and 'question' in value:
                                question_obj = value['question']
                                if isinstance(question_obj, dict) and 'question' in question_obj:
                                    question_text = question_obj['question'].strip()
                                    if question_text not in seen_question_texts:
                                        seen_question_texts.add(question_text)
                                        filtered_results[key] = value
                                    else:
                                        removed_count += 1
                                else:
                                    # Keep if question structure is different
                                    filtered_results[key] = value
                            else:
                                # Keep if structure is different
                                filtered_results[key] = value
                        
                        # Create new data structure with filtered results
                        new_data = data.copy()
                        new_data['results'] = filtered_results
                        
                        print(f"Removed {removed_count} duplicate questions.")
                        print(f"Kept {len(filtered_results)} unique questions.")
                        
                    elif isinstance(results, list):
                        # Structure with results list
                        seen_question_texts = set()
                        filtered_results = []
                        removed_count = 0
                        
                        for item in results:
                            question_text = None
                            if isinstance(item, dict):
                                if 'question' in item:
                                    if isinstance(item['question'], dict) and 'question' in item['question']:
                                        question_text = item['question']['question'].strip()
                                    elif isinstance(item['question'], str):
                                        question_text = item['question'].strip()
                                elif 'Question' in item:
                                    question_text = item['Question'].strip()
                            
                            if question_text:
                                if question_text not in seen_question_texts:
                                    seen_question_texts.add(question_text)
                                    filtered_results.append(item)
                                else:
                                    removed_count += 1
                            else:
                                # Keep if no question text found
                                filtered_results.append(item)
                        
                        new_data = data.copy()
                        new_data['results'] = filtered_results
                        
                        print(f"Removed {removed_count} duplicate questions.")
                        print(f"Kept {len(filtered_results)} unique questions.")
                    
                elif isinstance(data, list):
                    # Direct list of questions
                    seen_question_texts = set()
                    filtered_data = []
                    removed_count = 0
                    
                    for item in data:
                        if isinstance(item, dict):
                            question_text = None
                            if 'question' in item:
                                question_text = item['question'].strip() if isinstance(item['question'], str) else None
                            elif 'Question' in item:
                                question_text = item['Question'].strip() if isinstance(item['Question'], str) else None
                            
                            if question_text:
                                if question_text not in seen_question_texts:
                                    seen_question_texts.add(question_text)
                                    filtered_data.append(item)
                                else:
                                    removed_count += 1
                            else:
                                # Keep if no question text found
                                filtered_data.append(item)
                    
                    new_data = filtered_data
                    print(f"Removed {removed_count} duplicate questions.")
                    print(f"Kept {len(filtered_data)} unique questions.")
                else:
                    print(f"Error: Unsupported JSON structure")
                    return None
            elif isinstance(data, list):
                # Direct list
                seen_question_texts = set()
                filtered_data = []
                removed_count = 0
                
                for item in data:
                    if isinstance(item, dict):
                        question_text = None
                        if 'question' in item:
                            question_text = item['question'].strip() if isinstance(item['question'], str) else None
                        elif 'Question' in item:
                            question_text = item['Question'].strip() if isinstance(item['Question'], str) else None
                        
                        if question_text:
                            if question_text not in seen_question_texts:
                                seen_question_texts.add(question_text)
                                filtered_data.append(item)
                            else:
                                removed_count += 1
                        else:
                            filtered_data.append(item)
                
                new_data = filtered_data
                print(f"Removed {removed_count} duplicate questions.")
                print(f"Kept {len(filtered_data)} unique questions.")
            else:
                print(f"Error: Unsupported data structure: {type(data)}")
                return None
            
            # Write output JSON file
            print(f"Writing output to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            
            print(f"Output saved to: {output_path}")
            return output_path
            
        except json.JSONDecodeError:
            print("Error: Invalid JSON file")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    return output_path

def count_jsonl(file_path):
    """Count the number of questions in a JSONL file (one JSON object per line)."""
    try:
        print(f"Counting questions in JSONL file: {file_path}...")
        count = 0
        questions_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        questions_data.append(data)
                        count += 1
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines, found {count} questions so far...")
        print(f"\nTotal number of questions: {count}")
        
        # Check for duplicates
        if questions_data:
            check_duplicates_jsonl(questions_data, file_path)
        
        return count
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return 0

def count_unique_question_matches(jsonl_file_path, json_file_path):
    """Count unique question matches between a JSONL file and a JSON file.
    
    Args:
        jsonl_file_path: Path to JSONL file (e.g., data/PopMC_0_difficulty_filtered.jsonl)
        json_file_path: Path to JSON file (e.g., explicit_confidence_task_logs/...json)
    
    Returns:
        int: Number of unique question matches
    """
    if not os.path.exists(jsonl_file_path):
        print(f"Error: JSONL file not found: {jsonl_file_path}")
        return 0
    
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found: {json_file_path}")
        return 0
    
    print(f"Loading unique questions from {jsonl_file_path}...")
    # Extract unique questions from JSONL file
    jsonl_questions = set()
    jsonl_count = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    question_text = None
                    if 'question' in data:
                        question_text = data['question'].strip()
                    elif 'Question' in data:
                        question_text = data['Question'].strip()
                    
                    if question_text:
                        jsonl_questions.add(question_text)
                        jsonl_count += 1
                    
                    if line_num % 10000 == 0:
                        print(f"Processed {line_num} lines, found {len(jsonl_questions)} unique questions so far...")
                except json.JSONDecodeError:
                    continue
    
    print(f"Found {jsonl_count} total questions, {len(jsonl_questions)} unique questions in JSONL file")
    
    print(f"\nLoading unique questions from {json_file_path}...")
    # Extract unique questions from JSON file
    json_questions = set()
    json_count = 0
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle the results dictionary structure
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict) and 'question' in value:
                        question_obj = value['question']
                        if isinstance(question_obj, dict) and 'question' in question_obj:
                            question_text = question_obj['question'].strip()
                            if question_text:
                                json_questions.add(question_text)
                                json_count += 1
            elif isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        question_text = None
                        if 'question' in item:
                            if isinstance(item['question'], dict) and 'question' in item['question']:
                                question_text = item['question']['question'].strip()
                            elif isinstance(item['question'], str):
                                question_text = item['question'].strip()
                        elif 'Question' in item:
                            question_text = item['Question'].strip()
                        
                        if question_text:
                            json_questions.add(question_text)
                            json_count += 1
        elif isinstance(data, list):
            # Direct list format
            for item in data:
                if isinstance(item, dict):
                    question_text = None
                    if 'question' in item:
                        question_text = item['question'].strip() if isinstance(item['question'], str) else None
                    elif 'Question' in item:
                        question_text = item['Question'].strip() if isinstance(item['Question'], str) else None
                    
                    if question_text:
                        json_questions.add(question_text)
                        json_count += 1
        
        print(f"Found {json_count} total questions, {len(json_questions)} unique questions in JSON file")
        
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_file_path}")
        return 0
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return 0
    
    # Find intersection (unique matches)
    unique_matches = jsonl_questions & json_questions
    match_count = len(unique_matches)
    
    print(f"\n{'='*50}")
    print(f"Unique question matches: {match_count}")
    print(f"JSONL unique questions: {len(jsonl_questions)}")
    print(f"JSON unique questions: {len(json_questions)}")
    print(f"Match rate (matches / JSONL unique): {match_count / len(jsonl_questions) * 100:.2f}%")
    print(f"Match rate (matches / JSON unique): {match_count / len(json_questions) * 100:.2f}%")
    print(f"{'='*50}")
    
    return match_count

if __name__ == "__main__":
    file_path = "data/unfiltered-web-dev.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # count_questions(file_path)
    
    # # Also count questions in TriviaMC.jsonl and TriviaQA_fact_checked_rejections.jsonl
    # print("\n" + "="*50)
    # count_questions("data/TriviaMC.jsonl")
    
    # print("\n" + "="*50)
    # count_questions("data/TriviaQA_fact_checked_rejections.jsonl")
    
    # print("\n" + "="*50)
    # count_questions("capabilities_test_logs/llama-3.3-70b-instruct_TriviaMC_500_1762809440_test_data.json")
    
    # print("\n" + "="*50)
    # count_questions("data/PopMC_0_difficulty_filtered.jsonl")
    
    print("\n" + "="*50)
    count_questions("explicit_confidence_task_logs/llama-3.1-8b-instruct_PopMC_0_difficulty_filtered_11412_2025-11-25-17-02-17_explicit_confidence_task_all.json")
    
    print("\n" + "="*50)
    count_questions("explicit_confidence_task_logs/llama-3.1-8b-instruct_PopMC_0_difficulty_filtered_11412_2025-11-25-17-02-17_explicit_confidence_task_all_duplicate_questions_removed.json")
    
    print("\n" + "="*50)
    count_questions("data/PopMC.jsonl")
    
    print("\n" + "="*50)
    count_questions("data/PopMC_0_difficulty_filtered.jsonl")


    print("\n" + "="*50)
    count_questions("data/popqa_original.json")

    # Count unique matches between the two files
    print("\n" + "="*50)
    count_unique_question_matches(
        "data/PopMC_0_difficulty_filtered.jsonl",
        "explicit_confidence_task_logs/llama-3.1-8b-instruct_PopMC_0_difficulty_filtered_11412_2025-11-25-17-02-17_explicit_confidence_task_all_duplicate_questions_removed.json"
    )

    # remove_duplicate_questions('path/to/your/file.json')


