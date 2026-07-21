## Picking the right models for workflows and subagents

Rankings, higher = better. Cost reflects what I actually pay (OpenAI has really generous limits), not list price. Intelligence is how hard a problem you can hand the model unsupervised. Taste covers UI/UX, code quality, API design, and copy.

| model        | cost | intelligence | taste | speed |
|--------------|------|--------------|-------|-------|
| composer-2.5 | 9    | 7            | 4     | 9     |
| gpt-5.6      | 7    | 8            | 7     | 7     |
| sonnet-5     | 4    | 5            | 7     | 5     |
| opus-4.8     | 2    | 7            | 8     | 4     |
| fable-5      | 1    | 9            | 9     | 3     |

How to apply:
- These are defaults, not limits. You have standing permission to override them: if a cheaper model's output doesn't meet the bar, rerun or redo the work with a smarter model without asking. Judge the output, not the price tag. Escalating costs less than shipping mediocre work.
- Cost is a tie-breaker only; when axes conflict, intelligence > taste > cost > speed.
- Bulk work (clear-spec implementation, clear data analysis): composer-2.5 (more or less free)
- Mechanical work (semi independent implementation, slightly harder data analysis, migrations): gpt-5.6 (also more or less free, but not as much as composer)
- Anything user-facing (UI, copy, API design) needs taste ≥ 7.
- Reviews of plans/implementations: fable-5 or opus-4.8, optionally gpt-5.6 as an extra independent perspective.
- Never use Haiku.
- Mechanics: gpt is only reachable through the Codex CLI — `codex exec` / `codex review` (my `~/.codex/config.toml` defaults to gpt-5.6). composer-2.5 is reachable through the cursor cli - `agent -p "prompt" --model "composer-2.5"`
- Claude models (sonnet-5, opus-4.8, fable-5) run via the Agent/Workflow model parameter.

Using gpt-5.6 inside workflows and subagents (the model parameter only takes Claude models, so use a wrapper):
- Spawn a thin Claude wrapper agent with `model: 'sonnet', effort: 'low'` whose prompt instructs it to write a self-contained codex prompt, run `codex exec` via Bash, and return

Unless taste/judgement are highly important, don't use Anthropic models as subagents. Anthropic credits are a *large* limiter. GPT 5.6 will give you very similarly useful outputs. Default to the others unless doing ambiguous UX work; for rare/summary/end of session (aka "overall") judgement calls, you're free to use them, but the times that you'll need that will be rare and relatively explicit. 5.6 is a very good model; use it when you need intelligence most of the time. Composer is quite fast and capable, as well; use it as a workhorse.