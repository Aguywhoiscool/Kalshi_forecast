# Project authorization

The user authorizes routine commits and pushes of completed, reviewed work in
this repository to its configured origin, with at most 20 commits per calendar
day in America/New_York. This does not override sandbox or platform approvals.

Before each commit, inspect `git status`, review the staged diff, run appropriate
checks, and count today's distinct commits across all local refs by committer
timestamp. Conservatively count all authors toward the 20-commit cap; include
unreferenced commits created during the current session. Do not amend, rewrite
history, change dates or identities, or create other refs to evade the cap.
If the count is already 20, leave reviewed changes uncommitted until the next
day or an explicit user change to this limit. Push without force and do not
overwrite remote work. No automatic committing of unrelated changes.

Never commit credentials, collected data, model binaries, logs or generated
artifacts. Keep trading disabled. Do not modify or restart the collection
computer, evaluate reserved v2/v3 outcomes, or change Syncthing settings without
task-specific authorization. Dashboard work uses local read-only inference.
