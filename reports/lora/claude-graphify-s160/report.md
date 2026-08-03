# LoRA seed report

- model: `ibm-granite/granite-4.0-micro`
- examples: 42
- max_steps: 160
- elapsed_s: 244.018
- gpu_start: `{'cuda': True, 'name': 'Tesla T4', 'total_mib': 14912, 'allocated_mib': 6510, 'reserved_mib': 6512, 'max_allocated_mib': 6510}`
- gpu_end: `{'cuda': True, 'name': 'Tesla T4', 'total_mib': 14912, 'allocated_mib': 6566, 'reserved_mib': 8238, 'max_allocated_mib': 7134}`

## Eval Before

- task05_save: 1/5 `{'has_replace_between': False, 'has_save_marker': False, 'has_localstorage': False, 'has_check': False, 'no_wrapper': True}`
- commit_signature: 2/4 `{'has_trailer': False, 'has_git_log': False, 'no_push': True, 'no_gpg': True}`
- intent_question: 3/5 `{'mentions_options': False, 'asks_choice': False, 'no_git': True, 'no_pip': True, 'no_tool_call': True}`
- graphify_navigation: 2/5 `{'uses_graphify': False, 'mentions_graph': False, 'mentions_files': False, 'no_write_before_locating': True, 'no_git': True}`

## Eval After

- task05_save: 5/5 `{'has_replace_between': True, 'has_save_marker': True, 'has_localstorage': True, 'has_check': True, 'no_wrapper': True}`
- commit_signature: 2/4 `{'has_trailer': False, 'has_git_log': False, 'no_push': True, 'no_gpg': True}`
- intent_question: 3/5 `{'mentions_options': False, 'asks_choice': False, 'no_git': True, 'no_pip': True, 'no_tool_call': True}`
- graphify_navigation: 3/5 `{'uses_graphify': False, 'mentions_graph': True, 'mentions_files': False, 'no_write_before_locating': True, 'no_git': True}`
