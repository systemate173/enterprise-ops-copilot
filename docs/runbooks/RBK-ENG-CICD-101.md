# RBK-ENG-CICD-101 – CI/CD Deployment Failures

## Symptoms
- Application issues appear immediately after a deployment
- Errors may be browser- or environment-specific
- Rollbacks may temporarily resolve the issue

## Initial Checks
- Review the most recent deployment or release notes
- Check CI/CD pipeline logs for failures or warnings
- Confirm which environment is affected (prod/stage)

## Resolution Steps
- Roll back to the last known good deployment if needed
- Fix the identified regression and redeploy
- Validate behavior across supported browsers

## Escalation
- Escalate to the platform or frontend team if unresolved
