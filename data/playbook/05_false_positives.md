# False-Positive Guidance

## Large but legitimate payments
Customers occasionally make payments 5-12x their average: rent, school fees, electronics, travel. These are usually legitimate when:
- the device and city match the customer's usual ones,
- the payee is an existing contact or a low-risk merchant,
- there is no burst of other activity and no onward forwarding by the payee.

## Busy legitimate merchants
Popular low-risk merchants (grocery, fuel, pharmacy) naturally receive payments from many distinct payers. Fan-in on its own is not suspicious. Look for round amounts, a high risk score, and payouts.

## Decision rule
If the only signal is a high amount and every contextual check is normal, the verdict should be "legitimate" with action "no_action" or "monitor". Over-blocking good customers has a real cost and drives churn.
