# Architecture: plato-room-context

## Language Choice: Rust

### Why Rust

Context management is called on every message in every room. It must be:
- **Fast**: <0.1ms per context update
- **Predictable**: no GC pauses mid-conversation
- **Memory-efficient**: thousands of concurrent rooms

| Metric | Python (dict+deque) | Rust (VecDeque+struct) |
|--------|--------------------|-----------------------|
| Update 1K contexts | ~8ms | ~0.5ms |
| Memory per context | ~300 bytes | ~80 bytes |
| 10K concurrent rooms | ~3MB | ~800KB |

### Why not LangChain ConversationMemory

LangChain's memory abstractions are Python-only with dict-heavy storage.
No token budget management, no eviction policies, no priority system.
Our implementation adds: token-aware eviction, priority stacking, and importance scoring.

### Architecture

```
add(entry) → estimate_tokens() → push to VecDeque → maybe_evict()
                                                         ↓
                    EvictionPolicy { FIFO, LRU, Priority, Importance, Hybrid }
                                                         ↓
                    Remove lowest-score entry until within token budget
```

### Eviction Policies

1. **FIFO**: Oldest first (simple, predictable)
2. **LRU**: Same as FIFO for append-only (kept for API compatibility)
3. **Priority**: Lowest priority first (0=system=never, 3=tile=evict first)
4. **Importance**: Lowest importance score first (tiles with 0 importance go first)
5. **SlidingWindow**: Keep most recent N tokens (like a chat window)
6. **Hybrid**: Weighted combination of age + priority + importance (default)

### Token Budget

System prompts use reserved tokens (never evicted). User content uses the
remaining budget. When user tokens exceed budget, eviction kicks in.
