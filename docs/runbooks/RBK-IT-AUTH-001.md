# RBK-IT-AUTH-001 – Authentication / SSO Issues

## Symptoms
- Users unable to log in
- Authentication or token errors
- SSO redirects failing

## Initial Checks
- Check authentication service and IDP status
- Review recent configuration or certificate changes
- Validate token expiration and clock sync

## Resolution Steps
- Restart affected auth services if safe
- Roll back recent auth-related changes
- Clear caches and retry authentication

## Escalation
- Escalate to IAM or Security team if persistent
