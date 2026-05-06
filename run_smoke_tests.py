#!/usr/bin/env python
"""
Smoke tests runner for smtpBERT repository.
Runs: 1) unittest discover 2) compileall
"""
import sys
import subprocess
import os

os.chdir('d:\\smtpBERT')

print("=" * 60)
print("SMOKE TEST 1: Python Unittest Discovery")
print("=" * 60)

try:
    result1 = subprocess.run(
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
        capture_output=False,
        text=True
    )
    unittest_status = "PASSED" if result1.returncode == 0 else "FAILED"
    unittest_code = result1.returncode
except Exception as e:
    print(f"ERROR running unittest: {e}")
    unittest_status = "ERROR"
    unittest_code = 1

print(f"\nUnitTest Result: {unittest_status} (code: {unittest_code})")

print("\n" + "=" * 60)
print("SMOKE TEST 2: Python Compile All (src/)")
print("=" * 60)

try:
    result2 = subprocess.run(
        [sys.executable, '-m', 'compileall', 'src'],
        capture_output=False,
        text=True
    )
    compile_status = "PASSED" if result2.returncode == 0 else "FAILED"
    compile_code = result2.returncode
except Exception as e:
    print(f"ERROR running compileall: {e}")
    compile_status = "ERROR"
    compile_code = 1

print(f"\nCompile Result: {compile_status} (code: {compile_code})")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Unittest: {unittest_status}")
print(f"Compileall: {compile_status}")
sys.exit(max(unittest_code, compile_code))
