CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS accounts (
    account_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    home_city    TEXT NOT NULL,
    kyc_level    TEXT NOT NULL,          -- min | full
    opened_at    TIMESTAMPTZ NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active'   -- active | frozen
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    risk_score   REAL NOT NULL,          -- 0..1, from an imaginary risk vendor
    chargeback_rate REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id       TEXT PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL,
    payer_id     TEXT NOT NULL,
    payee_id     TEXT NOT NULL,
    merchant_id  TEXT,
    amount       NUMERIC(12,2) NOT NULL,
    channel      TEXT NOT NULL,          -- upi | card | wallet
    device_id    TEXT NOT NULL,
    city         TEXT NOT NULL,
    -- Ground truth from the generator. NEVER exposed to agent tools; used only by evals.
    is_fraud     BOOLEAN NOT NULL DEFAULT FALSE,
    fraud_pattern TEXT
);
CREATE INDEX IF NOT EXISTS idx_txn_payer_ts ON transactions (payer_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_txn_payee_ts ON transactions (payee_id, ts DESC);

CREATE TABLE IF NOT EXISTS flags (
    txn_id       TEXT PRIMARY KEY REFERENCES transactions(txn_id),
    score        REAL NOT NULL,
    reasons      JSONB NOT NULL,
    flagged_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cases (
    case_id      TEXT PRIMARY KEY,
    txn_id       TEXT NOT NULL REFERENCES transactions(txn_id),
    status       TEXT NOT NULL,          -- investigating | pending_review | closed
    severity     TEXT,
    case_file    JSONB,
    decision     TEXT,                   -- approved | rejected | auto_closed
    reviewer_note TEXT,
    cost_usd     REAL,
    latency_s    REAL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
