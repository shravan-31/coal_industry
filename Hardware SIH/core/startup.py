#!/usr/bin/env python3
"""
Startup script for the Enhanced R&D Proposal Evaluation System
"""
import argparse
import subprocess
import sys
import os

def run_streamlit():
    """Run the Streamlit web interface"""
    print("Starting Streamlit web interface...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", 
                   "--server.port=8501", "--server.address=0.0.0.0"])

def run_api():
    """Run the Flask API"""
    print("Starting Flask API...")
    subprocess.run([sys.executable, "api.py"])

def run_test():
    """Run the enhanced evaluator test"""
    print("Running enhanced evaluator test...")
    subprocess.run([sys.executable, "test_enhanced.py"])

def run_sample_data():
    """Generate sample data"""
    print("Generating sample data...")
    from enhanced_evaluator import create_sample_data
    create_sample_data()
    print("Sample data created successfully!")

def main():
    parser = argparse.ArgumentParser(description="Enhanced R&D Proposal Evaluation System")
    parser.add_argument("command", choices=["web", "api", "test", "sample"], 
                       help="Command to run: web (Streamlit), api (Flask), test (evaluation test), sample (generate sample data)")
    
    args = parser.parse_args()
    
    if args.command == "web":
        run_streamlit()
    elif args.command == "api":
        run_api()
    elif args.command == "test":
        run_test()
    elif args.command == "sample":
        run_sample_data()

if __name__ == "__main__":
    main()