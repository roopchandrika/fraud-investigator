# Money Mule Chains

## Typology
Stolen funds are moved quickly through a chain of mule accounts (layering) so the money is hard to recover. Each mule keeps a small cut (1-5%) and forwards the rest within minutes.

## Red flags
- An account receives a large amount and sends 80-100% of it onward within 20 minutes.
- Two or more consecutive pass-through hops.
- Mule accounts are typically younger than 60 days, have minimal KYC, and have little organic activity.
- The original sender is often a long-standing customer acting from a new device or city.

## Investigation steps
1. Trace the money forward from the first suspicious transfer.
2. Profile every account on the path (age, KYC, baseline activity).
3. Use the counterparty graph to see whether the mules also receive from other victims.

## Typical action
Freeze the mule account currently holding the funds (the last hop with a balance). If the origin transfer looks unauthorised, flag the victim account for customer contact. Freezing requires human approval.
