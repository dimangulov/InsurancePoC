-- Migration 001 — internationalise client_profiles
-- Run in Supabase SQL editor if you already applied schema.sql with the old columns.
-- Safe to run multiple times (uses IF EXISTS / IF NOT EXISTS guards).

ALTER TABLE client_profiles
    ADD COLUMN IF NOT EXISTS state_or_province TEXT,
    ADD COLUMN IF NOT EXISTS country           TEXT,
    ADD COLUMN IF NOT EXISTS postal_code       TEXT;

-- Copy existing data from old columns before dropping them
UPDATE client_profiles
SET    state_or_province = state,
       postal_code       = zip_code
WHERE  state_or_province IS NULL;

ALTER TABLE client_profiles
    DROP COLUMN IF EXISTS state,
    DROP COLUMN IF EXISTS zip_code;
