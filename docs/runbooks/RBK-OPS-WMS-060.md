# RBK-OPS-WMS-060 – Warehouse Management System (WMS) Outage

## Symptoms
- Orders not being picked, packed, or shipped
- Shipping labels fail to print
- Warehouse workflows are blocked or severely delayed
- Backlog of unfulfilled orders increases rapidly

## Initial Checks
- Confirm which warehouse locations are affected
- Check WMS application and database health
- Verify print servers, scanners, and network connectivity
- Review recent infrastructure or configuration changes

## Resolution Steps
- Restart affected WMS services if safe
- Restore connectivity to dependent systems (printers, scanners, network)
- Fail over to backup systems if available
- Manually process critical shipments if required

## Data to Collect
- Order IDs and timestamps for blocked shipments
- WMS service logs and error messages
- Printer/label service logs
- Recent infrastructure or network changes

## Escalation
- Escalate to Warehouse IT / Infrastructure teams
- Notify Operations leadership if fulfillment SLAs are at risk
