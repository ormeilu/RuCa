#!/usr/bin/env python
"""
Quick Start Guide for Running Tests

Запуск этого скрипта или выполнение команд в терминале
"""

import subprocess


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'=' * 70}")
    print(f"▶ {description}")
    print(f"{'=' * 70}")
    print(f"Command: {cmd}\n")

    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} - PASSED")
    except subprocess.CalledProcessError:
        print(f"❌ {description} - FAILED")
        return False
    return True


def main():
    """Run all test commands"""
    print("\n" + "=" * 70)
    print("RuCa Tools Testing - Quick Start")
    print("=" * 70)

    commands = [
        ("pytest src/ruca/tests/tools/ -v --tb=short", "Run all tests with short traceback"),
        ("pytest src/ruca/tests/tools/ --cov=ruca.tools --cov-report=term-missing", "Run tests with coverage report"),
        ("pytest src/ruca/tests/tools/test_weather_and_convert_tools.py -v", "Test Weather & Currency Tools"),
        ("pytest src/ruca/tests/tools/test_calculator_tools.py -v", "Test Calculator Tools"),
        ("pytest src/ruca/tests/tools/test_datetime_tools.py -v", "Test DateTime Tools"),
        ("pytest src/ruca/tests/tools/test_avia_tools.py -v", "Test Aviation Tools"),
        ("pytest src/ruca/tests/tools/test_retail_tools.py -v", "Test Retail/E-commerce Tools"),
        ("pytest src/ruca/tests/tools/test_translation_tools.py -v", "Test Translation Tools"),
    ]

    print("\nAvailable test commands:")
    for i, (cmd, desc) in enumerate(commands, 1):
        print(f"{i}. {desc}")
        print(f"   $ {cmd}\n")

    print("\nOther useful commands:")
    print("- Run tests with markers:")
    print("  $ pytest src/ruca/tests/tools/ -m unit")
    print("\n- Run tests and stop at first failure:")
    print("  $ pytest src/ruca/tests/tools/ -x")
    print("\n- Run specific test function:")
    print(
        "  $ pytest src/ruca/tests/tools/test_calculator_tools.py::TestCalculatorTool::test_safe_eval_simple_addition -v"
    )
    print("\n- Generate HTML coverage report:")
    print("  $ pytest src/ruca/tests/tools/ --cov=ruca.tools --cov-report=html")
    print("  (Open htmlcov/index.html in browser)")


if __name__ == "__main__":
    main()
