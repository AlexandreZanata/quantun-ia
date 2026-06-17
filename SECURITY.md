# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.9.x   | ✅        |
| < 0.9   | ❌        |

## Reporting a vulnerability

If you discover a security issue, please **do not** open a public GitHub issue.

1. Email the maintainer listed in [CITATION.cff](CITATION.cff) with subject `quantun-ia security`.
2. Include steps to reproduce, affected version, and impact assessment.
3. Allow up to 7 days for an initial response.

We will coordinate disclosure and credit researchers who report valid issues responsibly.

## Scope

In scope:

- REST API authentication (`JWT`, tenant isolation, async job queue)
- SQLite training job storage and multitenancy headers
- Dependency vulnerabilities flagged by `pip-audit` (see CI)

Out of scope:

- PennyLane / PyTorch upstream issues (report to those projects)
- Denial-of-service via expensive quantum training jobs on shared demo instances (mitigate with rate limits in production)

## Hardening notes for deployments

- Set `API_AUTH_REQUIRED=1` and rotate `API_AUTH_SECRET` / JWT keys in production.
- Never commit `.env`, JWT PEM files, or `logs/experiments.jsonl` with sensitive data.
- Use HTTPS termination in front of `qml-api`; the dev server is not production-hardened.
