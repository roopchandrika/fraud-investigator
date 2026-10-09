# Action Policy

## Actions available to the investigation agent
- **no_action**: the activity is explained and legitimate.
- **monitor**: the activity is unusual but the evidence is weak. Add the account to a 7-day watchlist.
- **block_transaction**: stop a single pending or reversible transaction.
- **freeze_account**: suspend all outgoing payments from an account.

## Human-in-the-loop requirement
The agent may never execute block_transaction or freeze_account by itself. Every such recommendation must be approved by a human analyst, who sees the case file and its cited evidence.

## Evidence standard
Every claim in a case file must cite the tool output it came from. Claims without a source are removed before review. A "fraud" verdict needs at least two independent pieces of evidence (for example behavioural baseline plus money flow).

## Which account to act on
Act on the account that stops further loss: the account holding the funds (mule chains, collusion cash-outs) or the compromised account (velocity bursts, takeovers).
