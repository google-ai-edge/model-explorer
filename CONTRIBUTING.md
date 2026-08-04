# Contributing to Model Explorer

Thank you for your interest in contributing!

## Security Requirements

To ensure the security of our CI/CD pipeline, we have integrated **zizmor**, a static analysis tool for GitHub Actions.

When contributing, please note the following new security requirements:
- All changes to GitHub Actions workflows (`.github/workflows/`) will be automatically scanned by `zizmor`.
- Ensure that your workflow changes pass all zizmor security checks. This includes avoiding untrusted input in `run` steps, pinning actions to specific commit SHAs, and adhering to the principle of least privilege for `GITHUB_TOKEN` permissions.
- Any modifications to GitHub Actions workflows or zizmor configurations require explicit approval from the security/infrastructure team.
