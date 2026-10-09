# Merchant Collusion and Cash-Out

## Typology
A high-risk merchant receives many round-amount card payments from fresh accounts (often funded with stolen money or used to launder it). It then pays out most of the collected money to one account within the hour.

## Red flags
- Five or more distinct payers within 30 minutes, mostly paying round amounts (1,000 / 2,000 / 5,000).
- Merchant risk score above 0.5 or chargeback rate above 2%.
- High-risk categories: gift cards, electronics resale, crypto OTC.
- A payout from the merchant to a single account equal to 80-95% of recent inflows.

## Investigation steps
1. Pull the merchant's risk profile and 24h inflow statistics.
2. Profile a sample of the payers (age, KYC).
3. Check where the merchant's payout went and that account's profile.

## Typical action
Block the payout transaction and escalate the merchant to the acquiring team. If the collector account is a likely mule, recommend freezing it. Either action requires human approval.
