# Current limitations

- Three-player Second Edition only. Historical protocol-2 APIs and checkpoints
  are intentionally unsupported.
- Visible history is explicit finite feedforward context, not a recurrent model
  or an oracle for current opponent hands. Its isolated playing-strength benefit
  has not been measured.
- The active model has one actor/critic architecture. Historical multihead and
  frozen-prior experiments are evidence about the retired system.
- Same-run snapshot opponents and repeatedly viewed development panels do not
  establish general playing strength or equilibrium behavior. Independent
  training seeds and opponent holdouts remain future work.
- Parallel collection improves throughput on the measured machine, but changes
  the per-game rollout horizon. It is not automatically a learning-quality gain.
- Resume starts fresh games at the last durable update, not exact midgame state.
  Dependencies have tested constraints rather than a complete transitive lock.
- File-based engine exchange and one engine process per game remain. GPU
  end-to-end throughput did not improve in the measured small-network setup.
- The dashboard reports aggregate metrics available from the active evaluator.
  Per-card opportunities, action timelines and age trajectories would require
  additional active telemetry; historical charts cannot be relabeled as new data.
- Supported development and process cleanup target Linux/WSL. Python 3.12 is the
  tested baseline although package metadata permits newer compatible Python.
