#!/usr/bin/env python3
"""
Script to count the number of questions in JSON or JSONL files
"""

import json
import sys
import os

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
                return count
            elif isinstance(data, dict):
                # If it's a dict, check for common keys that might contain the questions
                if 'Data' in data:
                    questions = data['Data']
                    if isinstance(questions, list):
                        count = len(questions)
                        print(f"\nTotal number of questions: {count}")
                        return count
                    else:
                        print(f"Error: 'Data' key contains {type(questions)}, expected a list")
                        return 0
                elif 'questions' in data:
                    count = len(data['questions'])
                    print(f"\nTotal number of questions: {count}")
                    return count
                elif 'results' in data:
                    count = len(data['results'])
                    print(f"\nTotal number of questions: {count}")
                    return count
                else:
                    print("Error: File contains a dictionary but no 'Data', 'questions', or 'results' key found.")
                    print(f"Available keys: {list(data.keys())}")
                    return 0
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

def count_jsonl(file_path):
    """Count the number of questions in a JSONL file (one JSON object per line)."""
    try:
        print(f"Counting questions in JSONL file: {file_path}...")
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines, found {count} questions so far...")
        print(f"\nTotal number of questions: {count}")
        return count
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return 0

if __name__ == "__main__":
    file_path = "data/unfiltered-web-dev.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    count_questions(file_path)
    
    # Also count questions in TriviaMC.jsonl and TriviaQA_fact_checked_rejections.jsonl
    print("\n" + "="*50)
    count_questions("data/TriviaMC.jsonl")
    
    print("\n" + "="*50)
    count_questions("data/TriviaQA_fact_checked_rejections.jsonl")
    
    print("\n" + "="*50)
    count_questions("capabilities_test_logs/llama-3.3-70b-instruct_TriviaMC_500_1762809440_test_data.json")
    
    print("\n" + "="*50)
    count_questions("data/PopMC_0_difficulty_filtered.jsonl")

