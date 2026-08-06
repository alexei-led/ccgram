# Contributing to CCGram

## Before You Write Code

Before you write any code, open a GitHub Issue.

This is not optional. An issue lets us agree on the approach before you write the implementation. A PR without a linked issue will be closed.

Write the actual problem in the issue. Include:

- What fails and why you think it is a bug
- What you want to add and why the project needs it
- The version of CCGram and the multiplexer (`tmux` or `herdr`)

Do not open an issue as a placeholder for a design you already decided on.

## Set Up Your Environment

1. Clone the repository.
2. Install the dependencies: `uv sync --extra dev`
3. Run the tests: `make test`
4. Run the linters: `make lint`

## Submit a Pull Request

1. Link the issue: write `Closes #<number>` in the PR description.
2. Change one thing per PR. Do not bundle unrelated fixes.
3. Add a test for each new behavior or bug fix.
4. Before you open the PR, make sure that `make test` and `make lint` pass.

CCGram targets Python 3.14. Match the style of the file you edit.

## What Gets Closed Without Review

A PR is closed without review if:

- There is no linked issue.
- It bundles unrelated changes.
- It adds a provider, a dependency, or a large feature without prior agreement in an issue.

A closed PR is not a rejection of the idea. Open or update the linked issue and continue from there.
