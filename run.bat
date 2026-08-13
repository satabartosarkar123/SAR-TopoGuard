@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
python run_overnight_pipeline.py --force_restart > pipeline_log.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL%
