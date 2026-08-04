# Token Logging Overhead

The accepted paired cohort used 24 requests and 3 paired repeats. Median wall-clock overhead was -0.017%, within the predeclared 1% limit.

Token timestamps were accumulated in memory and flushed once per block as compressed JSONL; there was no synchronous per-token disk I/O.
