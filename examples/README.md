# inferest examples

Runnable, offline demonstrations of the two questions inferest answers
best. Both scripts build the binary from source if needed and finish in
seconds.

| Script | Question it answers |
|---|---|
| `gpu-shortlist.sh` | "Which of these cards is actually worth buying for my target model?" — compares a GPU shortlist for an 8B and a 70B workload, then uses the `fit` exit-code gate the way a purchase checklist would. |
| `sbc-check.sh` | "Can a single-board computer run a small model usefully?" — fit + speed for 1B/3B classes on a Raspberry Pi 5 and a Jetson Orin Nano, plus a custom-device example for a board that is not in the preset table. |

Run them from anywhere:

```bash
bash examples/gpu-shortlist.sh
bash examples/sbc-check.sh
```

Both exercise only the public CLI — they double as living documentation of
the flags, and they are safe to re-run (no state, no network, no downloads).
