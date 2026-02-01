# RBK-OPS-LOGISTICS-050 – Inventory / Warehouse Sync Issues

## Symptoms
- Mismatch between website inventory and warehouse stock counts
- Customers can place orders for items that are actually out of stock
- Backorders, cancellations, or fulfillment delays increase

## Initial Checks
- Confirm scope: which SKUs, warehouses, regions, and time window
- Check status of inventory sync jobs (overnight batch or event-driven)
- Verify timestamps: last successful sync and last data update in each system
- Identify source of truth (WMS vs e-commerce platform vs ERP)

## Resolution Steps
- Re-run the failed sync job (manually or via scheduler) if safe
- Reconcile inventory for affected SKUs (spot-check counts and reservations)
- Validate that inventory updates propagate end-to-end (WMS → ERP → website)
- Add monitoring/alerts for job failures and data drift thresholds

## Data to Collect
- Example SKUs affected + expected vs actual counts
- Job run logs (success/failure, duration, error messages)
- Correlation IDs or batch IDs for the sync run
- Any recent changes to sync configuration, credentials, or endpoints

## Escalation
- Escalate to Operations Systems / Data Engineering if failures persist
- Escalate to Warehouse Ops if physical counts are suspected inaccurate
