#!/usr/bin/env python3
"""
tests/test_memory_chain.py
==========================

Quick functional test for session memory.
Simulates multi-turn conversation continuity.

Usage:
    python -m tests.test_memory_chain
"""

from cricket_tools.memory import clear_memory, merge_with_memory, update_memory, get_memory
import json, time

print("🔄 Starting memory chain test...\n")

# Step 1: fresh memory
clear_memory()
print("Step 1 — After clear:")
print(json.dumps(get_memory(), indent=2))

# Step 2: update with first query entities
update_memory({"team": "Mumbai Indians", "venue": "Wankhede Stadium"})
print("\nStep 2 — After adding MI + Wankhede:")
print(json.dumps(get_memory(), indent=2))

# Step 3: new query that omits team
merged = merge_with_memory({"venue": "Chepuk"})
print("\nStep 3 — Merge new (venue=Chepuk) with memory:")
print(json.dumps(merged, indent=2))

# Step 4: persist and reload
update_memory(merged)
print("\nStep 4 — After saving merged context:")
print(json.dumps(get_memory(), indent=2))

print("\n✅ Memory chain test complete.")