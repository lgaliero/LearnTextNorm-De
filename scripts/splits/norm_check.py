"""
NORM file validation and fixing utilities.
Ensures NORM files meet normEval.py requirements before inference.
"""

import os
import re
from typing import List, Tuple, Optional, Dict


class NormValidationError(Exception):
    """Raised when NORM file has unfixable validation errors."""
    pass


def check_norm_file(file_path: str, verbose: bool = True) -> Dict[str, List[Tuple[int, str]]]:
    """
    Check NORM file for formatting issues.
    
    Args:
        file_path: Path to .norm file
        verbose: Print detailed issue information
        
    Returns:
        Dictionary mapping issue type to list of (line_number, line_content) tuples
        
    Issue types:
        - 'multi_column': Lines with >2 tab-separated columns
        - 'space_separated': Lines with spaces instead of tabs
        - 'multi_tab': Lines with multiple consecutive tabs
        - 'whitespace_only': Lines with only whitespace (should be empty)
        - 'single_column': Non-empty lines with only one column
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    issues = {
        'multi_column': [],
        'space_separated': [],
        'multi_tab': [],
        'whitespace_only': [],
        'single_column': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_no_newline = line.rstrip('\n')
            
            # Skip truly empty lines (sentence separators)
            if line_no_newline == '':
                continue
            
            # Check 1: Lines with more than 2 tab-separated columns
            tabs = line_no_newline.split('\t')
            if len(tabs) > 2:
                issues['multi_column'].append((line_num, line_no_newline))
            
            # Check 2: Lines with spaces instead of tabs (single column with spaces)
            if '\t' not in line_no_newline and ' ' in line_no_newline and line_no_newline.strip():
                issues['space_separated'].append((line_num, line_no_newline))
            
            # Check 3: Multiple consecutive tabs
            if '\t\t' in line_no_newline:
                issues['multi_tab'].append((line_num, line_no_newline))
            
            # Check 4: Whitespace-only lines (should be completely empty)
            if line_no_newline != line_no_newline.strip() and not line_no_newline.strip():
                issues['whitespace_only'].append((line_num, line_no_newline))
            
            # Check 5: Single column (potential missing tab)
            if len(tabs) == 1 and line_no_newline.strip():
                issues['single_column'].append((line_num, line_no_newline))
    
    # Print report if verbose
    if verbose:
        total_issues = sum(len(v) for v in issues.values())
        
        if total_issues == 0:
            print(f"✓ {file_path}: No issues found")
        else:
            print(f"\n{'='*80}")
            print(f"VALIDATION REPORT: {file_path}")
            print(f"{'='*80}")
            
            if issues['multi_column']:
                print(f"\n⚠️  Lines with >2 columns: {len(issues['multi_column'])}")
                for line_num, line in issues['multi_column'][:5]:
                    print(f"  Line {line_num}: {line[:70]}...")
                if len(issues['multi_column']) > 5:
                    print(f"  ... and {len(issues['multi_column']) - 5} more")
            
            if issues['space_separated']:
                print(f"\n⚠️  Space-separated lines: {len(issues['space_separated'])}")
                for line_num, line in issues['space_separated'][:5]:
                    print(f"  Line {line_num}: {line[:70]}...")
                if len(issues['space_separated']) > 5:
                    print(f"  ... and {len(issues['space_separated']) - 5} more")
            
            if issues['multi_tab']:
                print(f"\n⚠️  Multiple consecutive tabs: {len(issues['multi_tab'])}")
                for line_num, line in issues['multi_tab'][:5]:
                    print(f"  Line {line_num}: {repr(line[:70])}...")
                if len(issues['multi_tab']) > 5:
                    print(f"  ... and {len(issues['multi_tab']) - 5} more")
            
            if issues['whitespace_only']:
                print(f"\n⚠️  Whitespace-only lines: {len(issues['whitespace_only'])}")
                for line_num, line in issues['whitespace_only'][:5]:
                    print(f"  Line {line_num}: {repr(line)}")
                if len(issues['whitespace_only']) > 5:
                    print(f"  ... and {len(issues['whitespace_only']) - 5} more")
            
            if issues['single_column']:
                print(f"\n⚠️  Single-column lines: {len(issues['single_column'])}")
                for line_num, line in issues['single_column'][:5]:
                    print(f"  Line {line_num}: {line[:70]}...")
                if len(issues['single_column']) > 5:
                    print(f"  ... and {len(issues['single_column']) - 5} more")
            
            print(f"\n{'='*80}")
            print(f"Total issues: {total_issues}")
            print(f"{'='*80}")
    
    return issues


def fix_norm_file(file_path: str, output_path: Optional[str] = None, 
                  backup: bool = True, verbose: bool = True) -> str:
    """
    Fix formatting issues in NORM file.
    
    Args:
        file_path: Path to .norm file to fix
        output_path: Output path (if None, overwrites original)
        backup: Create .bak backup before fixing
        verbose: Print detailed fix information
        
    Returns:
        Path to fixed file
        
    Fixes applied:
        1. Multi-column: Merge all-but-last as source, last as target
        2. Space-separated: Convert whitespace to single tab
        3. Multi-tab: Replace multiple tabs with single tab
        4. Whitespace-only: Convert to empty lines
        5. Single-column: Leave as-is (may be intentional)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine output path
    if output_path is None:
        output_path = file_path
    
    # Create backup if requested
    if backup and output_path == file_path:
        backup_path = file_path + '.bak'
        with open(file_path, 'r', encoding='utf-8') as src, \
             open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        if verbose:
            print(f"✓ Created backup: {backup_path}")
    
    fixed_lines = []
    fixes_applied = {
        'multi_column': 0,
        'space_separated': 0,
        'multi_tab': 0,
        'whitespace_only': 0
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_no_newline = line.rstrip('\n')
            
            # Truly empty lines - preserve as-is
            if line_no_newline == '':
                fixed_lines.append('')
                continue
            
            # Fix 1: Lines with more than 2 columns
            tabs = line_no_newline.split('\t')
            if len(tabs) > 2:
                # Merge all-but-last as source, last as target
                source = ' '.join(tabs[:-1])
                target = tabs[-1]
                fixed_lines.append(f"{source}\t{target}")
                fixes_applied['multi_column'] += 1
                continue
            
            # Fix 2: Space-separated lines (convert to tab-separated)
            if '\t' not in line_no_newline and ' ' in line_no_newline and line_no_newline.strip():
                parts = line_no_newline.split()
                if len(parts) >= 2:
                    # Take first part as source, rest as target
                    source = parts[0]
                    target = ' '.join(parts[1:])
                    fixed_lines.append(f"{source}\t{target}")
                    fixes_applied['space_separated'] += 1
                else:
                    fixed_lines.append(line_no_newline)
                continue
            
            # Fix 3: Multiple consecutive tabs
            if '\t\t' in line_no_newline:
                fixed_line = re.sub(r'\t+', '\t', line_no_newline)
                fixed_lines.append(fixed_line)
                fixes_applied['multi_tab'] += 1
                continue
            
            # Fix 4: Whitespace-only lines (should be empty)
            if line_no_newline != line_no_newline.strip() and not line_no_newline.strip():
                fixed_lines.append('')
                fixes_applied['whitespace_only'] += 1
                continue
            
            # No issues - keep as-is
            fixed_lines.append(line_no_newline)
    
    # Write fixed file
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    if verbose:
        total_fixes = sum(fixes_applied.values())
        if total_fixes > 0:
            print(f"\n{'='*80}")
            print(f"FIXES APPLIED: {output_path}")
            print(f"{'='*80}")
            
            if fixes_applied['multi_column']:
                print(f"  ✓ Multi-column lines fixed: {fixes_applied['multi_column']}")
            if fixes_applied['space_separated']:
                print(f"  ✓ Space-separated lines fixed: {fixes_applied['space_separated']}")
            if fixes_applied['multi_tab']:
                print(f"  ✓ Multiple tabs fixed: {fixes_applied['multi_tab']}")
            if fixes_applied['whitespace_only']:
                print(f"  ✓ Whitespace-only lines fixed: {fixes_applied['whitespace_only']}")
            
            print(f"\nTotal fixes: {total_fixes}")
            print(f"{'='*80}")
        else:
            print(f"✓ No fixes needed: {output_path}")
    
    return output_path


def validate_and_fix_norm_files(
    directory: str,
    fix: bool = False,
    backup: bool = True,
    verbose: bool = True,
    recursive: bool = False
) -> Dict[str, Dict]:
    """
    Validate (and optionally fix) all NORM files in a directory.
    
    Args:
        directory: Directory containing .norm files
        fix: Apply fixes to files
        backup: Create backups before fixing
        verbose: Print detailed information
        recursive: Search subdirectories for .norm files
        
    Returns:
        Dictionary mapping filenames to their issue reports
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all .norm files
    if recursive:
        norm_files = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith('.norm'):
                    norm_files.append(os.path.join(root, f))
    else:
        norm_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.norm')]
    
    if not norm_files:
        print(f"No .norm files found in {directory}")
        return {}
    
    print(f"\n{'='*80}")
    print(f"VALIDATING {len(norm_files)} NORM FILE(S) IN: {directory}")
    if recursive:
        print("(searching subdirectories)")
    print(f"{'='*80}")
    
    results = {}
    
    for file_path in sorted(norm_files):
        filename = os.path.basename(file_path)
        relative_path = os.path.relpath(file_path, directory)
        
        if verbose:
            print(f"\n--- {relative_path} ---")
        
        # Check for issues
        issues = check_norm_file(file_path, verbose=verbose)
        results[relative_path] = issues
        
        # Apply fixes if requested
        if fix:
            total_issues = sum(len(v) for v in issues.values())
            if total_issues > 0:
                fix_norm_file(file_path, backup=backup, verbose=verbose)
                
                # Re-check after fixing
                if verbose:
                    print("\nRe-checking after fixes...")
                    new_issues = check_norm_file(file_path, verbose=True)
                    remaining = sum(len(v) for v in new_issues.values())
                    if remaining == 0:
                        print("✓ All issues resolved")
                    else:
                        print(f"⚠️  {remaining} issues remaining (may need manual review)")
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    files_with_issues = sum(1 for issues in results.values() if sum(len(v) for v in issues.values()) > 0)
    
    if files_with_issues == 0:
        print("✓ All NORM files are properly formatted")
    else:
        print(f"⚠️  {files_with_issues}/{len(norm_files)} files have issues")
        
        if fix:
            print("\n✓ Fixes have been applied")
            print("  Please re-run validation to confirm all issues are resolved")
        else:
            print("\n  Run with --fix flag to automatically fix these issues")
    
    print(f"{'='*80}\n")
    
    return results


def batch_validate_from_config(
    paths_config: Optional[Dict] = None,
    fix: bool = False,
    backup: bool = True,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Batch validate all NORM files using paths from config.
    
    Args:
        paths_config: Dict with TRAIN_NORM, DEV_NORM, TEST_NORM paths
        fix: Apply fixes to files
        backup: Create backups before fixing
        verbose: Print detailed information
        
    Returns:
        Dictionary mapping filenames to their issue reports
    """
    # Get norm file paths from config
    norm_paths = []
    missing_files = []
    undefined_paths = []
    
    if not paths_config:
        print("❌ Error: paths_config is None or empty")
        return {}
    
    for key in ['TRAIN_NORM', 'DEV_NORM', 'TEST_NORM']:
        path = paths_config.get(key)
        
        if not path:
            undefined_paths.append(key)
        elif not os.path.exists(path):
            missing_files.append((key, path))
        else:
            norm_paths.append(path)
    
    # Report issues
    if undefined_paths:
        print(f"⚠️  Warning: These paths are not defined in config:")
        for key in undefined_paths:
            print(f"     - {key}")
    
    if missing_files:
        print(f"\n⚠️  Warning: These NORM files don't exist yet:")
        for key, path in missing_files:
            print(f"     - {key}: {path}")
        print("\n💡 Tip: Create NORM files first with: python data_maker.py norm")
    
    if not norm_paths:
        print("\n❌ No NORM files found to validate.")
        print("\nPossible solutions:")
        print("1. Create NORM files: python data_maker.py norm")
        print("2. Use directory mode: python data_maker.py validate --directory ../data --recursive")
        return {}
    
    print(f"\n{'='*80}")
    print(f"BATCH VALIDATION: {len(norm_paths)} NORM FILE(S) FROM CONFIG")
    print(f"{'='*80}")
    
    results = {}
    
    for file_path in sorted(norm_paths):
        filename = os.path.basename(file_path)
        
        if verbose:
            print(f"\n--- {filename} ({file_path}) ---")
        
        # Check for issues
        issues = check_norm_file(file_path, verbose=verbose)
        results[filename] = issues
        
        # Apply fixes if requested
        if fix:
            total_issues = sum(len(v) for v in issues.values())
            if total_issues > 0:
                fix_norm_file(file_path, backup=backup, verbose=verbose)
                
                # Re-check after fixing
                if verbose:
                    print("\nRe-checking after fixes...")
                    new_issues = check_norm_file(file_path, verbose=True)
                    remaining = sum(len(v) for v in new_issues.values())
                    if remaining == 0:
                        print("✓ All issues resolved")
                    else:
                        print(f"⚠️  {remaining} issues remaining (may need manual review)")
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    files_with_issues = sum(1 for issues in results.values() if sum(len(v) for v in issues.values()) > 0)
    
    if files_with_issues == 0:
        print("✓ All NORM files are properly formatted")
    else:
        print(f"⚠️  {files_with_issues}/{len(norm_paths)} files have issues")
        
        if fix:
            print("\n✓ Fixes have been applied")
            print("  Please re-run validation to confirm all issues are resolved")
        else:
            print("\n  Run with validation command and --fix flag to automatically fix these issues")
    
    print(f"{'='*80}\n")
    
    return results


def get_norm_statistics(file_path: str) -> Dict[str, int]:
    """
    Get statistics about a NORM file.
    
    Args:
        file_path: Path to .norm file
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_lines': 0,
        'empty_lines': 0,
        'word_pairs': 0,
        'single_column': 0,
        'multi_column': 0,
        'sentences': 0
    }
    
    consecutive_empty = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stats['total_lines'] += 1
            line_no_newline = line.rstrip('\n')
            
            if line_no_newline == '':
                stats['empty_lines'] += 1
                consecutive_empty += 1
                if consecutive_empty == 1:
                    stats['sentences'] += 1
            else:
                consecutive_empty = 0
                tabs = line_no_newline.split('\t')
                
                if len(tabs) == 2:
                    stats['word_pairs'] += 1
                elif len(tabs) == 1:
                    stats['single_column'] += 1
                else:
                    stats['multi_column'] += 1
    
    return stats