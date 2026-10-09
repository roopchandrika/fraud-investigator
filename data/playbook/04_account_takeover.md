# Account Takeover (ATO)

## Typology
An attacker gains access to a customer's credentials and makes a single large transfer from a new device, often from a different city, to a beneficiary the customer has never paid.

## Red flags
- New device AND new city in the same session.
- An amount 5x or more above the customer's 30-day average.
- A beneficiary that is new to the account, young, and has minimal KYC.

## Investigation steps
1. Confirm the device and city are new against the 30-day baseline.
2. Profile the beneficiary.
3. Check whether the beneficiary forwards the funds onward.

## Typical action
Block the transaction if it is still reversible and freeze the account pending customer verification. This requires human approval.
