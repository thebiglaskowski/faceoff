"""
Simple test to verify terminal tab logging works.
Run this to test the terminal tab functionality.
"""

import logging
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.components.terminal_tab import (
    get_terminal_output,
    get_terminal_stats,
    terminal_handler,
)


def test_terminal_logging():
    """Test terminal logging functionality."""

    # Set up basic logging
    logging.basicConfig(level=logging.DEBUG)

    # Add our terminal handler to root logger
    root_logger = logging.getLogger()

    # Remove any existing terminal handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        if hasattr(handler, "log_buffer"):
            root_logger.removeHandler(handler)

    # Add our terminal handler
    root_logger.addHandler(terminal_handler)

    print("🧪 Testing Terminal Tab Logging...")
    print("=" * 50)

    # Create test logger
    test_logger = logging.getLogger("TerminalTest")

    # Send various log levels
    test_logger.debug("🔍 This is a DEBUG message")
    test_logger.info("📘 This is an INFO message")
    test_logger.warning("⚠️ This is a WARNING message")
    test_logger.error("❌ This is an ERROR message")
    test_logger.critical("💥 This is a CRITICAL message")

    # Wait a moment for processing
    time.sleep(0.1)

    # Check results
    print("\n📊 Terminal Stats:")
    print(get_terminal_stats())

    print("\n🖥️ Terminal Output:")
    output = get_terminal_output()
    print(output)

    # Check if it worked
    logs = terminal_handler.get_logs()
    if logs:
        print(f"\n✅ SUCCESS! Terminal captured {len(logs)} log entries")
        return True
    else:
        print("\n❌ FAILED! No logs captured")
        return False


if __name__ == "__main__":
    success = test_terminal_logging()
    print(f"\nTest Result: {'PASSED' if success else 'FAILED'}")
