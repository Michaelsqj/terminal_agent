# Judge Prompt Versioning

This directory contains versioned judge prompts for evaluating terminal agent performance.

## Structure

- `terminal_agent_judge_prompt_v{X.Y}.md` - Versioned prompt files
- `terminal_agent_judge_prompt_latest.md` - Symlink to the current latest version
- `versions.json` - Version registry with metadata

## Using a Specific Version

### In Code
```python
from src.tbench_rllm.rewards.judge_reward import calculate_judge_score

# Use latest version (default)
score = await calculate_judge_score(messages, dockerfile_contents)

# Use specific version
score = await calculate_judge_score(messages, dockerfile_contents, judge_version="v4.0")
```

### In Judge Evaluation Script
```bash
# Use latest version (default)
python judge_eval.py --model "n"

# Use specific version
python judge_eval.py --model "n" --judge-version "v4.0"
```

## Adding a New Version

1. Create a new prompt file: `terminal_agent_judge_prompt_v{X.Y}.md`
2. Update `versions.json` with the new version info
3. Update the `latest` field in `versions.json` if this is the new default
4. Update the symlink: `ln -sf terminal_agent_judge_prompt_v{X.Y}.md terminal_agent_judge_prompt_latest.md`

## Version History

See `versions.json` for detailed version history and changes.