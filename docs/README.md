# Project documentation

The supported system is the three-player **Second Edition (2020), protocol 3**
agent with explicit own-hand and visible-history inputs. Legacy trainers and
their runtime compatibility are retired.

| Task | Read |
| --- | --- |
| Start, stop or resume training | [Training](../TRAINING.md) |
| Install, develop and choose appropriate checks | [Development](guides/DEVELOPMENT.md) |
| Understand PPO, collection, league and checkpoint boundaries | [Architecture](reference/ARCHITECTURE.md) |
| See exactly what the agent may observe | [Observation contract](reference/DECISION_OBSERVATION_CONTRACT.md) |
| Evaluate a saved snapshot and export its dashboard | [Evaluation](guides/EVALUATION.md) |
| Update the separate engine | [Engine workflow](guides/ENGINE.md) |
| Interpret completed measurements | [Results](results/README.md) |
| Understand current limits | [Known issues](reference/KNOWN_ISSUES.md) |
| Find research questions and the archived literature review | [Research](research/AGENDA.md) |
| Review the consolidation | [Cleanup record](reference/CLEANUP.md) |

Current behavior has one description: operation in guides, contracts in reference,
results in results. Historical source and full experiment reports remain available
through immutable Git links. Generated runs, weights and routine check logs stay local.
