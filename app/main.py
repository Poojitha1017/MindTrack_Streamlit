# main.py
import argparse
import json
import os
import pandas as pd

# Import from app package
from app.baseline import train_user_baseline, train_from_file
from app.detection import detect_and_interpret
from app.feedback import collect_feedback
from app.visualization import visualize_baseline
from app.utils import load_df
from app.config import MODELS_DIR


def cmd_train(args):
    if args.usercol:
        train_from_file(
            args.data,
            usercol=args.usercol,
            outdir=args.models_dir,
            contamination=args.contamination
        )
    else:
        df = load_df(args.data)
        train_user_baseline(
            df,
            user_id=args.user_id,
            outdir=args.models_dir,
            contamination=args.contamination
        )


def cmd_detect(args):
    with open(args.entry_json, "r", encoding="utf-8") as f:
        entry = json.load(f)
    out = detect_and_interpret(entry, user_id=args.user_id, outdir=args.models_dir)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_visualize(args):
    df = load_df(args.data)
    new_entry = None
    if args.entry_json and os.path.exists(args.entry_json):
        with open(args.entry_json, "r", encoding="utf-8") as f:
            new_entry = json.load(f)
    visualize_baseline(
        df=df,
        user_id=args.user_id,
        outdir=args.models_dir,
        new_entry=new_entry,
        save_plots=args.save_plots
    )


def cmd_feedback(args):
    with open(args.entry_json, "r", encoding="utf-8") as f:
        entry = json.load(f)
    out = detect_and_interpret(entry, user_id=args.user_id, outdir=args.models_dir)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    collect_feedback(out, feedback_store=args.store)


def build_parser():
    p = argparse.ArgumentParser(prog="MindTrack", description="MindTrack CLI")
    p.add_argument("--models_dir", default=MODELS_DIR, help="Directory to store/load models")

    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Train baseline")
    t.add_argument("--data", required=True, help="CSV/XLS(X) dataset path")
    t.add_argument("--usercol", default=None, help="User ID column (if multi-user file)")
    t.add_argument("--user_id", default="user123", help="User ID (single-user data)")
    t.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination")
    t.set_defaults(func=cmd_train)

    # detect
    d = sub.add_parser("detect", help="Detect anomaly for a new entry JSON")
    d.add_argument("--user_id", required=True)
    d.add_argument("--entry_json", required=True, help="Path to JSON with new-entry features")
    d.set_defaults(func=cmd_detect)

    # visualize
    v = sub.add_parser("visualize", help="Visualize baseline + optional new entry")
    v.add_argument("--user_id", required=True)
    v.add_argument("--data", required=True, help="CSV/XLS(X) dataset path")
    v.add_argument("--entry_json", default=None, help="Optional JSON with new-entry features")
    v.add_argument("--save_plots", action="store_true", help="Also save plots to models_dir")
    v.set_defaults(func=cmd_visualize)

    # feedback
    f = sub.add_parser("feedback", help="Run detection and log user feedback")
    f.add_argument("--user_id", required=True)
    f.add_argument("--entry_json", required=True, help="Path to JSON with new-entry features")
    f.add_argument("--store", default="user_feedback.json", help="Where to store feedback log")
    f.set_defaults(func=cmd_feedback)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
