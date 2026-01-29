"""Pretty printer for context assembly logs.

Reads log lines from stdin and formats them with colors.
Usage: tail -f logs/context_assembly.log | python scripts/pretty_log.py
"""

import re
import sys


def colorize(line: str) -> str:
    """Add ANSI colors to log line based on content."""
    
    # Timestamp
    line = re.sub(
        r'(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])',
        r'\033[90m\1\033[0m',  # Gray
        line
    )
    
    # Component tags
    line = re.sub(
        r'\[(ContextEngine|RAG|Memory|Skill|LLM)\]',
        lambda m: f'\033[1;36m[{m.group(1)}]\033[0m',  # Cyan bold
        line
    )
    
    # RAG algorithms
    line = re.sub(
        r'\[(BM25|HNSW|IVF_PQ|RFF)\]',
        lambda m: f'\033[1;35m[{m.group(1)}]\033[0m',  # Magenta bold
        line
    )
    
    # Numbers/stats
    line = re.sub(
        r'(\d+\.?\d*\s*(tokens?|ms|chunks?|results?|%)|top[_-]?\d+|confidence:\s*\d+\.\d+)',
        r'\033[33m\1\033[0m',  # Yellow
        line,
        flags=re.IGNORECASE
    )
    
    # Success/Error
    line = line.replace("success", "\033[32msuccess\033[0m")
    line = line.replace("error", "\033[31merror\033[0m")
    line = line.replace("✓", "\033[32m✓\033[0m")
    line = line.replace("✅", "\033[32m✅\033[0m")
    line = line.replace("❌", "\033[31m❌\033[0m")
    
    return line


def main():
    """Read from stdin and print colored output."""
    print("\033[90mContext Assembly Log Monitor started...\033[0m")
    print("\033[90m-" * 60 + "\033[0m")
    
    try:
        for line in sys.stdin:
            print(colorize(line), end="")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\033[90mMonitor stopped.\033[0m")


if __name__ == "__main__":
    main()
