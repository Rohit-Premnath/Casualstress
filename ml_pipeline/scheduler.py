"""
Real-Time Data Pipeline Scheduler
====================================
Automates daily data pulls, processing, and regime detection.

Schedule:
- 6:00 AM ET: Pull FRED data
- 6:30 AM ET: Pull Yahoo Finance data
- 7:00 AM ET: Run data quality checks
- 7:30 AM ET: Re-process data and re-detect regime
- Sunday 2:00 AM ET: Re-run causal discovery (weekly)

Also monitors for regime changes and logs alerts.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("CausalStress.Scheduler")


# ============================================
# DATABASE
# ============================================

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================
# JOB 1: FRED DATA PULL (6:00 AM ET)
# ============================================

def job_fred_pull():
    """Pull latest FRED data."""
    logger.info("=" * 50)
    logger.info("JOB: FRED Data Pull - Starting")
    try:
        from data_ingestion.fred_fetcher import main as fred_main
        fred_main()
        logger.info("JOB: FRED Data Pull - SUCCESS")
        log_job_result("fred_pull", "success")
    except Exception as e:
        logger.error(f"JOB: FRED Data Pull - FAILED: {e}")
        log_job_result("fred_pull", "failed", str(e))


# ============================================
# JOB 2: YAHOO FINANCE PULL (6:30 AM ET)
# ============================================

def job_yahoo_pull():
    """Pull latest Yahoo Finance data."""
    logger.info("=" * 50)
    logger.info("JOB: Yahoo Finance Pull - Starting")
    try:
        from data_ingestion.yahoo_fetcher import main as yahoo_main
        yahoo_main()
        logger.info("JOB: Yahoo Finance Pull - SUCCESS")
        log_job_result("yahoo_pull", "success")
    except Exception as e:
        logger.error(f"JOB: Yahoo Finance Pull - FAILED: {e}")
        log_job_result("yahoo_pull", "failed", str(e))


# ============================================
# JOB 3: DATA QUALITY CHECK (7:00 AM ET)
# ============================================

def job_data_quality():
    """Run data quality checks on the latest data."""
    logger.info("=" * 50)
    logger.info("JOB: Data Quality Check - Starting")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        issues = []

        # Check 1: FRED data freshness
        cursor.execute("""
            SELECT series_id, MAX(date) as latest
            FROM raw_fred.observations
            GROUP BY series_id
        """)
        fred_latest = cursor.fetchall()
        today = datetime.now().date()
        for series_id, latest_date in fred_latest:
            days_stale = (today - latest_date).days
            if days_stale > 7:
                issues.append(f"FRED {series_id}: {days_stale} days stale (last: {latest_date})")

        # Check 2: Yahoo data freshness
        cursor.execute("""
            SELECT ticker, MAX(date) as latest
            FROM raw_yahoo.daily_prices
            GROUP BY ticker
        """)
        yahoo_latest = cursor.fetchall()
        for ticker, latest_date in yahoo_latest:
            days_stale = (today - latest_date).days
            if days_stale > 3:  # Market data should be at most 3 days old (weekends)
                issues.append(f"Yahoo {ticker}: {days_stale} days stale (last: {latest_date})")

        # Check 3: Row count sanity
        cursor.execute("SELECT COUNT(*) FROM raw_fred.observations")
        fred_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM raw_yahoo.daily_prices")
        yahoo_count = cursor.fetchone()[0]

        if fred_count < 20000:
            issues.append(f"FRED row count suspiciously low: {fred_count}")
        if yahoo_count < 80000:
            issues.append(f"Yahoo row count suspiciously low: {yahoo_count}")

        cursor.close()
        conn.close()

        if issues:
            logger.warning(f"Data quality issues found ({len(issues)}):")
            for issue in issues:
                logger.warning(f"  - {issue}")
            log_job_result("data_quality", "warning", json.dumps(issues))
        else:
            logger.info(f"JOB: Data Quality Check - ALL CLEAR (FRED: {fred_count} rows, Yahoo: {yahoo_count} rows)")
            log_job_result("data_quality", "success")

    except Exception as e:
        logger.error(f"JOB: Data Quality Check - FAILED: {e}")
        log_job_result("data_quality", "failed", str(e))


# ============================================
# JOB 4: DATA PROCESSING + REGIME DETECTION (7:30 AM ET)
# ============================================

def job_process_and_detect():
    """Re-process data and re-detect current regime."""
    logger.info("=" * 50)
    logger.info("JOB: Data Processing + Regime Detection - Starting")
    try:
        # Step 1: Re-process data
        logger.info("  Step 1: Processing data...")
        from data_ingestion.data_processor import main as processor_main
        processor_main()

        # Step 2: Re-detect regime
        logger.info("  Step 2: Detecting regime...")
        from regime_detection.hmm_model import main as hmm_main
        hmm_main()

        # Step 3: Check for regime change
        check_regime_change()

        logger.info("JOB: Data Processing + Regime Detection - SUCCESS")
        log_job_result("process_and_detect", "success")

    except Exception as e:
        logger.error(f"JOB: Data Processing + Regime Detection - FAILED: {e}")
        log_job_result("process_and_detect", "failed", str(e))


# ============================================
# JOB 5: WEEKLY CAUSAL GRAPH RERUN (Sunday 2:00 AM ET)
# ============================================

def job_weekly_causal():
    """Re-run causal discovery weekly with latest data."""
    logger.info("=" * 50)
    logger.info("JOB: Weekly Causal Discovery - Starting")
    try:
        from causal_discovery.dynotears_engine import main as dyno_main
        dyno_main()
        logger.info("JOB: Weekly Causal Discovery - SUCCESS")
        log_job_result("weekly_causal", "success")
    except Exception as e:
        logger.error(f"JOB: Weekly Causal Discovery - FAILED: {e}")
        log_job_result("weekly_causal", "failed", str(e))


# ============================================
# REGIME CHANGE DETECTION
# ============================================

def check_regime_change():
    """
    Check if the regime changed from yesterday.
    If so, log an alert event for notification.
    """
    logger.info("  Checking for regime change...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get last 2 regime classifications
    cursor.execute("""
        SELECT date, regime_name, probability
        FROM models.regimes
        ORDER BY date DESC
        LIMIT 2
    """)

    rows = cursor.fetchall()

    if len(rows) < 2:
        logger.info("  Not enough regime history to compare")
        cursor.close()
        conn.close()
        return

    today_regime = rows[0]
    yesterday_regime = rows[1]

    if today_regime[1] != yesterday_regime[1]:
        # REGIME CHANGE DETECTED!
        old_regime = yesterday_regime[1]
        new_regime = today_regime[1]
        confidence = today_regime[2]

        logger.warning(f"  !!! REGIME CHANGE DETECTED !!!")
        logger.warning(f"  {old_regime.upper()} -> {new_regime.upper()} (confidence: {confidence:.1%})")

        # Store alert in database
        cursor.execute("""
            INSERT INTO app.regime_alerts (id, date, old_regime, new_regime, confidence, notified)
            VALUES (gen_random_uuid(), %s, %s, %s, %s, FALSE)
        """, (today_regime[0], old_regime, new_regime, confidence))

        conn.commit()
        log_job_result("regime_change", "alert",
                      json.dumps({"from": old_regime, "to": new_regime, "confidence": confidence}))
    else:
        logger.info(f"  No regime change. Still in {today_regime[1]} ({today_regime[2]:.1%})")

    cursor.close()
    conn.close()


# ============================================
# JOB LOGGING
# ============================================

def log_job_result(job_name, status, details=None):
    """Log job execution result to database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO app.scheduler_log (id, job_name, status, details, executed_at)
            VALUES (gen_random_uuid(), %s, %s, %s, NOW())
        """, (job_name, status, details))

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log job result: {e}")


# ============================================
# DATABASE SETUP (create tables if missing)
# ============================================

def setup_scheduler_tables():
    """Create tables needed by the scheduler if they don't exist."""
    logger.info("Setting up scheduler tables...")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app.scheduler_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_name VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            details TEXT,
            executed_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app.regime_alerts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            date DATE NOT NULL,
            old_regime VARCHAR(30),
            new_regime VARCHAR(30),
            confidence DOUBLE PRECISION,
            notified BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()

    logger.info("  Scheduler tables ready")


# ============================================
# MANUAL RUN (run all jobs once, right now)
# ============================================

def run_all_now():
    """Run all jobs immediately (for testing)."""
    logger.info("=" * 60)
    logger.info("RUNNING ALL JOBS MANUALLY")
    logger.info("=" * 60)

    setup_scheduler_tables()

    logger.info("\n[1/5] FRED Pull...")
    job_fred_pull()

    logger.info("\n[2/5] Yahoo Finance Pull...")
    job_yahoo_pull()

    logger.info("\n[3/5] Data Quality Check...")
    job_data_quality()

    logger.info("\n[4/5] Data Processing + Regime Detection...")
    job_process_and_detect()

    logger.info("\n[5/5] Causal Discovery (skipping in manual mode - takes too long)...")
    logger.info("  Run separately: python -m causal_discovery.dynotears_engine")

    logger.info("\n" + "=" * 60)
    logger.info("ALL JOBS COMPLETE")
    logger.info("=" * 60)


# ============================================
# SCHEDULER SETUP
# ============================================

def start_scheduler():
    """Start the APScheduler with all cron jobs."""
    logger.info("=" * 60)
    logger.info("CAUSALSTRESS - REAL-TIME DATA PIPELINE")
    logger.info("=" * 60)

    setup_scheduler_tables()

    scheduler = BlockingScheduler(timezone="US/Eastern")

    # Job 1: FRED pull at 6:00 AM ET, Monday-Friday
    scheduler.add_job(
        job_fred_pull,
        CronTrigger(hour=6, minute=0, day_of_week="mon-fri"),
        id="fred_pull",
        name="FRED Data Pull",
        misfire_grace_time=3600,
    )

    # Job 2: Yahoo Finance pull at 6:30 AM ET, Monday-Friday
    scheduler.add_job(
        job_yahoo_pull,
        CronTrigger(hour=6, minute=30, day_of_week="mon-fri"),
        id="yahoo_pull",
        name="Yahoo Finance Pull",
        misfire_grace_time=3600,
    )

    # Job 3: Data quality check at 7:00 AM ET, Monday-Friday
    scheduler.add_job(
        job_data_quality,
        CronTrigger(hour=7, minute=0, day_of_week="mon-fri"),
        id="data_quality",
        name="Data Quality Check",
        misfire_grace_time=3600,
    )

    # Job 4: Process + regime detection at 7:30 AM ET, Monday-Friday
    scheduler.add_job(
        job_process_and_detect,
        CronTrigger(hour=7, minute=30, day_of_week="mon-fri"),
        id="process_detect",
        name="Data Processing + Regime Detection",
        misfire_grace_time=3600,
    )

    # Job 5: Weekly causal discovery at 2:00 AM ET, Sunday
    scheduler.add_job(
        job_weekly_causal,
        CronTrigger(hour=2, minute=0, day_of_week="sun"),
        id="weekly_causal",
        name="Weekly Causal Discovery",
        misfire_grace_time=7200,
    )

    logger.info("Scheduled jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")

    logger.info("\nScheduler starting... (Press Ctrl+C to stop)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CausalStress Real-Time Pipeline")
    parser.add_argument("--mode", choices=["schedule", "run-now", "test-quality"],
                       default="run-now",
                       help="schedule=start cron scheduler, run-now=run all jobs once, test-quality=run quality check only")

    args = parser.parse_args()

    if args.mode == "schedule":
        start_scheduler()
    elif args.mode == "run-now":
        run_all_now()
    elif args.mode == "test-quality":
        setup_scheduler_tables()
        job_data_quality()