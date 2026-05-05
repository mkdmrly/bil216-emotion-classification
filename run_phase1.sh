#!/bin/zsh
set -e

. .venv/bin/activate
python3 train.py --data-dir data/raw --model auto
python3 generate_phase1_report.py

echo
echo "Done. Check:"
echo "  results/metrics.json"
echo "  results/classification_report.txt"
echo "  results/confusion_matrix.png"
echo "  submission/FinalProject_GROUP18_Phase1_Report.pdf"
