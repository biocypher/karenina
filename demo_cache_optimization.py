#!/usr/bin/env python3
"""Demonstration of enhanced answer cache optimization.

This script shows the cache optimization in action with:
- Non-blocking cache returns
- Task shuffling
- Progressive retry strategy
- Result order preservation
"""

import random
import threading
import time
from karenina.utils.answer_cache import AnswerTraceCache

print("=" * 70)
print("ENHANCED ANSWER CACHE OPTIMIZATION DEMONSTRATION")
print("=" * 70)

# Demo 1: Non-blocking cache
print("\n📋 Demo 1: Non-blocking Cache Behavior")
print("-" * 70)

cache = AnswerTraceCache()
key = "test_answer_key"

# Worker 1: Reserve and generate
print("Worker 1: Reserving slot...")
status1, _ = cache.get_or_reserve(key)
print(f"  → Status: {status1}")
assert status1 == "MISS"

# Worker 2: Check immediately (should not block)
print("\nWorker 2: Checking cache (should return immediately)...")
start = time.time()
status2, _ = cache.get_or_reserve(key)
elapsed = time.time() - start
print(f"  → Status: {status2} (elapsed: {elapsed*1000:.1f}ms)")
assert status2 == "IN_PROGRESS"
assert elapsed < 0.01, "Should return immediately!"

# Complete the answer
print("\nWorker 1: Completing answer generation...")
cache.complete(key, {"answer": "42"})

# Worker 3: Should hit cache
print("Worker 3: Checking cache...")
status3, data = cache.get_or_reserve(key)
print(f"  → Status: {status3}, Data: {data}")
assert status3 == "HIT"
assert data == {"answer": "42"}

print("\n✓ Non-blocking behavior verified!\n")

# Demo 2: Parallel access with progressive retry
print("📋 Demo 2: Progressive Retry Strategy")
print("-" * 70)

cache2 = AnswerTraceCache()
key2 = "slow_answer"

# Simulate slow generation
generation_complete = threading.Event()

def slow_generator():
    status, _ = cache2.get_or_reserve(key2)
    print(f"Generator: Got status {status}, starting generation...")
    time.sleep(1.0)  # Simulate slow generation
    cache2.complete(key2, {"result": "generated"})
    generation_complete.set()
    print("Generator: Complete!")

def fast_checker(worker_id):
    # First check - immediate IN_PROGRESS
    start = time.time()
    status, data = cache2.get_or_reserve(key2)
    elapsed = time.time() - start
    print(f"Checker {worker_id}: First check → {status} ({elapsed*1000:.0f}ms)")

    if status == "IN_PROGRESS":
        # In real implementation, would requeue here
        # For demo, wait and check again
        print(f"Checker {worker_id}: Would requeue and move to next task...")
        time.sleep(0.3)  # Simulate doing other work

        # Second check after generation
        generation_complete.wait()
        status, data = cache2.get_or_reserve(key2)
        print(f"Checker {worker_id}: Second check → {status}, got result!")

# Start threads
print("Starting parallel workers...")
gen_thread = threading.Thread(target=slow_generator)
check_threads = [threading.Thread(target=fast_checker, args=(i,)) for i in range(2)]

gen_thread.start()
time.sleep(0.1)  # Let generator reserve first

for t in check_threads:
    t.start()

gen_thread.join()
for t in check_threads:
    t.join()

print("\n✓ Progressive retry demonstrated!\n")

# Demo 3: Cache statistics
print("📋 Demo 3: Cache Statistics")
print("-" * 70)

cache3 = AnswerTraceCache()

print("Performing various cache operations...")
# 3 misses
for i in range(3):
    status, _ = cache3.get_or_reserve(f"key_{i}")
    print(f"  Operation {i+1}: {status}")
    cache3.complete(f"key_{i}", {"data": f"value_{i}"})

# 2 hits
for i in range(2):
    status, _ = cache3.get_or_reserve(f"key_{i}")
    print(f"  Operation {i+4}: {status}")

# 1 IN_PROGRESS
cache3.get_or_reserve("key_100")
status, _ = cache3.get_or_reserve("key_100")
print(f"  Operation 6: {status}")

stats = cache3.get_stats()
print(f"\n📊 Final Statistics:")
print(f"  Hits: {stats['hits']}")
print(f"  Misses: {stats['misses']}")
print(f"  IN_PROGRESS encounters: {stats['waits']}")

assert stats['hits'] == 2
assert stats['misses'] == 4  # 3 initial + 1 from key_100 reservation
assert stats['waits'] == 1

print("\n✓ Statistics tracking verified!\n")

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The enhanced cache optimization provides:
✓ Non-blocking returns - Workers never wait idle
✓ Progressive retry - Immediate requeue first, then 30s waits
✓ Thread-safe operation - Multiple workers can access safely
✓ Accurate statistics - Track hits, misses, and IN_PROGRESS

Benefits:
• Better thread utilization - No 5-minute blocking waits
• Higher throughput - Workers always have work to do
• Simpler timeout logic - 30s instead of 5 minutes
• Guaranteed completion - No max retry limits
""")

print("🎉 All demonstrations passed successfully!")
print("=" * 70)
