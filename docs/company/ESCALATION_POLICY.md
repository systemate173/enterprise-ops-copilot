# ESCALATION_POLICY

## When to escalate
- High urgency incidents: notify on-call owner immediately, then escalate if not resolved quickly
- Customer-facing payment failures: escalate to Payments Engineering; loop in Finance if refunds/charges involved
- Auth/VPN access failures: escalate to IAM/Security if persistent or widespread
- Deploy-related regressions: escalate to Platform; involve owning product team (Frontend/Mobile) as needed
- Performance degradation (multi-second load times / elevated latency): escalate to SRE

## Escalation format
- Escalation target should be a TEAM name (not a sentence)
- Include: start time, scope, error text/logs (if available), suspected system, and what actions were taken