# LoRA seed report

- model: `ibm-granite/granite-4.0-micro`
- examples: 23
- max_steps: 80
- elapsed_s: 266.533
- gpu_start: `{'cuda': True, 'name': 'Tesla T4', 'total_mib': 14912, 'allocated_mib': 6510, 'reserved_mib': 6512, 'max_allocated_mib': 6510}`
- gpu_end: `{'cuda': True, 'name': 'Tesla T4', 'total_mib': 14912, 'allocated_mib': 6566, 'reserved_mib': 7962, 'max_allocated_mib': 7134}`

## Eval Before

- task05_save: 1/5 `{'has_replace_between': False, 'has_save_marker': False, 'has_localstorage': False, 'has_check': False, 'no_wrapper': True}`
- commit_workflow: 7/9 `{'has_git_commit': True, 'mentions_reason': True, 'has_git_status': True, 'has_git_log': False, 'valid_tool_calls': False, 'no_unknown_tools': True, 'no_push': True, 'no_gpg': True, 'no_text_signature_default': True}`
- commit_literal_signature: 5/7 `{'has_git_commit': True, 'has_text_signature': True, 'has_git_log': False, 'valid_tool_calls': False, 'no_unknown_tools': True, 'no_push': True, 'no_gpg': True}`
- intent_question: 3/5 `{'mentions_options': False, 'asks_choice': False, 'no_git': True, 'no_pip': True, 'no_tool_call': True}`
- graphify_navigation: 3/7 `{'uses_graphify': False, 'mentions_graph': False, 'mentions_files': False, 'valid_tool_calls': False, 'no_unknown_tools': True, 'no_write_before_locating': True, 'no_git': True}`

## Eval After

- task05_save: 5/5 `{'has_replace_between': True, 'has_save_marker': True, 'has_localstorage': True, 'has_check': True, 'no_wrapper': True}`
- commit_workflow: 4/9 `{'has_git_commit': False, 'mentions_reason': False, 'has_git_status': False, 'has_git_log': False, 'valid_tool_calls': False, 'no_unknown_tools': True, 'no_push': True, 'no_gpg': True, 'no_text_signature_default': True}`
- commit_literal_signature: 4/7 `{'has_git_commit': False, 'has_text_signature': True, 'has_git_log': False, 'valid_tool_calls': True, 'no_unknown_tools': False, 'no_push': True, 'no_gpg': True}`
- intent_question: 4/5 `{'mentions_options': False, 'asks_choice': True, 'no_git': True, 'no_pip': True, 'no_tool_call': True}`
- graphify_navigation: 3/7 `{'uses_graphify': False, 'mentions_graph': False, 'mentions_files': False, 'valid_tool_calls': True, 'no_unknown_tools': False, 'no_write_before_locating': True, 'no_git': True}`
