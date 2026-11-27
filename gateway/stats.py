"""Counters for GET /stats. Kept in process, because the gateway runs as a single process
(Prometheus is a stretch goal). Only the event loop calls record, so no lock is needed."""


class Stats:
    # HIT-VERIFIED comes in Task 9; listed now so /stats keeps the same shape.
    RESULTS = ("BYPASS", "HIT-EXACT", "HIT-SEMANTIC", "HIT-VERIFIED", "MISS")

    def __init__(self):
        self.counts = {result: 0 for result in self.RESULTS}
        self.tokens_saved = 0
        self.upstream_ms_saved = 0.0

    def record(self, result: str, tokens_saved: int = 0, upstream_ms_saved: float = 0.0) -> None:
        self.counts[result] += 1
        self.tokens_saved += tokens_saved
        self.upstream_ms_saved += upstream_ms_saved

    def snapshot(self, entry_count: int | None) -> dict:
        hits = self.counts["HIT-EXACT"] + self.counts["HIT-SEMANTIC"] + self.counts["HIT-VERIFIED"]
        # BYPASS is left out: those requests were never allowed to use the cache.
        lookups = hits + self.counts["MISS"]
        return {
            "counts": dict(self.counts),
            "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
            "tokens_saved": self.tokens_saved,
            # The upstream time the served entries originally took: an estimate of GPU time saved.
            "gpu_seconds_saved": round(self.upstream_ms_saved / 1000, 2),
            "entries": entry_count,
        }
