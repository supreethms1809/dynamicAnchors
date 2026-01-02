#!/usr/bin/env python3
"""
Parse test_rules log file and find the best rule for each class.

The best rule is defined as the rule that has the highest number of samples
from that class satisfying the rule (highest class-conditional coverage).
"""

import re
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def parse_log_file(log_file: str) -> Dict[int, List[Dict]]:
    """
    Parse the log file and extract rule information.
    
    Returns:
        Dictionary mapping class_id to list of rules with their metrics.
        Each rule dict contains:
        - rule_num: Rule number (1-indexed)
        - rule_desc: Rule description
        - rollout_type: 'instance_based' or 'class_based'
        - source_classes: List of source classes
        - samples_satisfying: Number of samples from this class satisfying the rule
        - coverage: Coverage percentage (as float, e.g., 48.00 for 48%)
    """
    rules_by_class = defaultdict(list)
    current_rule = None
    current_class = None
    
    def save_current_rule():
        """Helper function to save the current rule to rules_by_class"""
        nonlocal current_rule, current_class
        if current_rule and current_rule['class_metrics']:
            # Save the rule for all classes it covers
            for class_id, metrics in current_rule['class_metrics'].items():
                rule_entry = {
                    'rule_num': current_rule['rule_num'],
                    'rule_desc': current_rule['rule_desc'],
                    'rollout_type': current_rule['rollout_type'],
                    'source_classes': current_rule['source_classes'],
                    'samples_satisfying': metrics['samples_satisfying'],
                    'coverage': metrics['coverage']
                }
                rules_by_class[class_id].append(rule_entry)
    
    # Patterns
    rule_pattern = re.compile(r'Rule (\d+)/\d+: (.+?)$')
    rollout_type_pattern = re.compile(r'Rollout type: (\w+)')
    source_classes_pattern = re.compile(r'Source classes: \[(.*?)\]')
    class_header_pattern = re.compile(r'Class (\d+):')
    samples_pattern = re.compile(r'Samples satisfying: (\d+)/\d+ \((\d+\.\d+)% coverage\)')
    separator_pattern = re.compile(r'^=+$')  # Line with only = characters
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check if this is a rule header - if so, save previous rule first
            rule_match = rule_pattern.search(line)
            if rule_match:
                # Save previous rule if it exists
                save_current_rule()
                
                # Start new rule
                rule_num = int(rule_match.group(1))
                rule_desc = rule_match.group(2).strip()
                current_rule = {
                    'rule_num': rule_num,
                    'rule_desc': rule_desc,
                    'rollout_type': None,
                    'source_classes': None,
                    'class_metrics': {}  # class_id -> {samples_satisfying, coverage}
                }
                current_class = None
                continue
            
            if not current_rule:
                continue  # Skip lines until we find a rule
            
            # Match rollout type
            if current_rule['rollout_type'] is None:
                rollout_match = rollout_type_pattern.search(line)
                if rollout_match:
                    current_rule['rollout_type'] = rollout_match.group(1)
                    continue
            
            # Match source classes
            if current_rule['source_classes'] is None:
                source_match = source_classes_pattern.search(line)
                if source_match:
                    source_str = source_match.group(1).strip()
                    if source_str:
                        current_rule['source_classes'] = [int(x.strip()) for x in source_str.split(',')]
                    else:
                        current_rule['source_classes'] = []
                    continue
            
            # Match class header
            class_match = class_header_pattern.search(line)
            if class_match:
                current_class = int(class_match.group(1))
                continue
            
            # Match samples satisfying (only for the current class)
            if current_class is not None:
                samples_match = samples_pattern.search(line)
                if samples_match:
                    n_samples = int(samples_match.group(1))
                    coverage = float(samples_match.group(2))
                    current_rule['class_metrics'][current_class] = {
                        'samples_satisfying': n_samples,
                        'coverage': coverage
                    }
                    continue
    
    # Save the last rule
    save_current_rule()
    
    return dict(rules_by_class)


def find_best_rule_for_class(rules: List[Dict]) -> Optional[Dict]:
    """
    Find the best rule for a class (highest number of samples satisfying).
    
    Args:
        rules: List of rule dictionaries for a class
        
    Returns:
        The rule dictionary with the highest samples_satisfying, or None if empty
    """
    if not rules:
        return None
    
    best_rule = max(rules, key=lambda r: r['samples_satisfying'])
    return best_rule


def main():
    parser = argparse.ArgumentParser(
        description='Find the best rule for each class from test_rules log file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python find_best_rules.py test_rules_20260101_023340.log
    
Output:
    For each class (0, 1, 2), prints the rule with the highest number of
    samples from that class satisfying the rule.
        """
    )
    parser.add_argument(
        'log_file',
        type=str,
        help='Path to test_rules log file'
    )
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help='Classes to analyze (default: 0 1 2)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=1,
        help='Number of top rules to show per class (default: 1)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Parsing log file:", args.log_file)
    print("=" * 80)
    print("\nNote: Rule numbers in the log are based on alphabetical sorting of rule strings,")
    print("not on extraction order. Use the full rule description to uniquely identify rules.")
    
    # Parse the log file
    rules_by_class = parse_log_file(args.log_file)
    
    print(f"\nFound rules for classes: {sorted(rules_by_class.keys())}")
    for class_id in sorted(rules_by_class.keys()):
        print(f"  Class {class_id}: {len(rules_by_class[class_id])} rules")
    
    print("\n" + "=" * 80)
    print("BEST RULES BY CLASS")
    print("=" * 80)
    
    # Find best rules for each class
    for class_id in sorted(args.classes):
        if class_id not in rules_by_class:
            print(f"\nClass {class_id}: No rules found")
            continue
        
        rules = rules_by_class[class_id]
        
        if args.top_n == 1:
            best_rule = find_best_rule_for_class(rules)
            if best_rule:
                print(f"\n{'='*80}")
                print(f"Class {class_id} - Best Rule:")
                print(f"{'='*80}")
                print(f"Rule Number: {best_rule['rule_num']} (in alphabetical order)")
                print(f"Description: {best_rule['rule_desc']}")
                if best_rule['rule_desc'].endswith('...'):
                    print(f"  (Note: '...' indicates truncation in log - full rule may be longer)")
                print(f"Rollout Type: {best_rule['rollout_type']}")
                print(f"Source Classes: {best_rule['source_classes']}")
                print(f"Samples Satisfying: {best_rule['samples_satisfying']}/50")
                print(f"Coverage: {best_rule['coverage']:.2f}%")
        else:
            # Sort by samples_satisfying (descending) and take top N
            sorted_rules = sorted(rules, key=lambda r: r['samples_satisfying'], reverse=True)
            top_rules = sorted_rules[:args.top_n]
            
            print(f"\n{'='*80}")
            print(f"Class {class_id} - Top {len(top_rules)} Rules:")
            print(f"{'='*80}")
            for i, rule in enumerate(top_rules, 1):
                print(f"\nRank {i}:")
                print(f"  Rule Number: {rule['rule_num']} (in alphabetical order)")
                print(f"  Description: {rule['rule_desc']}")
                if rule['rule_desc'].endswith('...'):
                    print(f"    (Note: '...' indicates truncation in log - full rule may be longer)")
                print(f"  Rollout Type: {rule['rollout_type']}")
                print(f"  Source Classes: {rule['source_classes']}")
                print(f"  Samples Satisfying: {rule['samples_satisfying']}/50")
                print(f"  Coverage: {rule['coverage']:.2f}%")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    for class_id in sorted(args.classes):
        if class_id in rules_by_class:
            rules = rules_by_class[class_id]
            best_rule = find_best_rule_for_class(rules)
            if best_rule:
                print(f"Class {class_id}: Rule {best_rule['rule_num']} - {best_rule['samples_satisfying']}/50 samples ({best_rule['coverage']:.2f}% coverage)")


if __name__ == "__main__":
    main()

