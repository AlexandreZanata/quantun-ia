CREATE TABLE IF NOT EXISTS training_jobs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    profile TEXT NOT NULL,
    exp_id TEXT NOT NULL DEFAULT 'nano_train',
    seed INTEGER,
    epochs INTEGER,
    status TEXT NOT NULL,
    result_json TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_tenant_id
    ON training_jobs (tenant_id);

CREATE INDEX IF NOT EXISTS idx_training_jobs_tenant_status
    ON training_jobs (tenant_id, status);
