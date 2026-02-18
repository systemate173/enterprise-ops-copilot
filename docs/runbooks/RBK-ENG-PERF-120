# RBK-ENG-PERF-120 — Web Performance Degradation

## Symptoms
- Pages load slowly (multi-second load times)
- Elevated latency / timeouts
- Intermittent 5xx errors (503/504)
- Reports of “slow checkout” that appear system-wide

## Initial Checks
- Confirm scope: which endpoints/pages, region, and environment (prod/stage)
- Check APM metrics: p95/p99 latency, error rate, saturation (CPU/memory), and throughput
- Check dependency health: database latency, cache hit rate, queue depth
- Identify recent changes: deployments, feature flags, infra changes

## Triage Steps
1. Compare latency before/after last deploy
2. Inspect slow queries / DB wait events
3. Check autoscaling / capacity and resource saturation
4. Validate CDN/cache behavior (cache bypass, origin load)
5. Check timeouts between services (gateway/load balancer/app)

## Mitigation
- Roll back recent deploy / disable feature flag if correlated
- Increase capacity temporarily (scale up/out)
- Apply hotfix for obvious N+1 or expensive query regression
- Adjust timeouts only if justified and safe

## Escalation
- If customer impact is broad or errors are high: escalate to On-Call + SRE
- If DB latency is high: involve Database/Infra team
