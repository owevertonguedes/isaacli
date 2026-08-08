# LoRA seed report

- model: `ibm-granite/granite-4.0-micro`
- examples: 18
- max_steps: 3
- elapsed_s: 168.962
- gpu_start: `{'cuda': True, 'name': 'Tesla T4', 'total_mib': 14912, 'allocated_mib': 6510, 'reserved_mib': 6512, 'max_allocated_mib': 6510}`
- gpu_end: `{'cuda': True, 'name': 'Tesla T4', 'total_mib': 14912, 'allocated_mib': 6566, 'reserved_mib': 7156, 'max_allocated_mib': 6873}`

## Eval Before

- task05_save: 1/5 `{'has_replace_between': False, 'has_save_marker': False, 'has_localstorage': False, 'has_check': False, 'no_wrapper': True}`
- commit_signature: 2/4 `{'has_trailer': False, 'has_git_log': False, 'no_push': True, 'no_gpg': True}`
- intent_question: 3/5 `{'mentions_options': False, 'asks_choice': False, 'no_git': True, 'no_pip': True, 'no_tool_call': True}`

## Eval After

- task05_save: 1/5 `{'has_replace_between': False, 'has_save_marker': False, 'has_localstorage': False, 'has_check': False, 'no_wrapper': True}`
- commit_signature: 2/4 `{'has_trailer': False, 'has_git_log': False, 'no_push': True, 'no_gpg': True}`
- intent_question: 3/5 `{'mentions_options': False, 'asks_choice': False, 'no_git': True, 'no_pip': True, 'no_tool_call': True}`
