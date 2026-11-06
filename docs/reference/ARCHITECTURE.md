# Active architecture

The RL package supports the pinned three-player Second Edition engine only.
`SevenWondersEnv` aliases `DecisionEnv`. Historical protocol-2 models and APIs
are no longer supported; their source remains in Git history.

1. `DecisionEnv` owns a standard-library engine client, seat-relative observations
   and separate memory for every seat. It advances opponents until the learner
   has an actual decision. Follow-up discards and eligible seventh cards are real
   steps; skipped turns do not create samples.
2. The observation includes the own complete hand, public table, offered options
   and ordered legally observed history. Actor and critic see the same information.
   Opponent private hands and transport identities are excluded. See the
   [observation contract](DECISION_OBSERVATION_CONTRACT.md).
3. `DecisionActorCritic` encodes current public/hand data and ordered padded
   histories, scores offered candidates and estimates value from shared context.
   The action mask excludes unavailable options. There are up to 128 options;
   action meaning is bound to a physical card, decision ID and offered revision.
4. One PPO trainer handles every worker count. Local collection supports a single
   environment and explicit callbacks; process collection owns multiple engines
   and frozen opponents and shares observations with the learner. Arbitrary Python
   opponent callbacks require one worker. Only the parent updates the learner,
   selects league pairs, records results and saves the durable checkpoint.
5. GAE is computed independently for each game's trajectory before minibatching.
   Defaults: gamma .99, lambda .95, PPO clip .2, Adam learning rate .0003, four
   epochs, minibatches of at most 16 and gradient clipping at .5. Terminal reward
   is the VP margin to the strongest rival. Failed/unapplied exchanges cannot
   become samples. Four workers share the total rollout size; they do not each
   collect an entire rollout.

The league publishes frozen inference snapshots, initially at step zero and then
every 65,536 decisions. Ordered left/right opponent pairs are sampled using a
PFSP-inspired rule over discounted terminal win-share results, with a uniform
control fraction and neutral prior. These estimates mix changing learners and
are not fixed-policy payoffs or a multiplayer equilibrium solution. Evaluation
results never enter league feedback.

`checkpoint.pt` atomically stores model, optimizer, Torch RNG, feature contract,
configuration and league state. Immutable snapshot hashes, journal lengths and
reserved episode seeds support repair after interrupted writes. A directory lock
rejects concurrent trainers. Resume restores the durable update, repairs journal
tails and starts fresh games; abandoned games are not scored as losses. Current
`decision-ppo-v2` checkpoints and the observation/model architecture remain supported.

Training manifests record source hashes, Git/worktree state, binary and engine
commit, seed plan, dependency versions and configuration. These identify a run,
but do not guarantee bitwise reproducibility across platforms or dependency changes.

The independent evaluator copies committed snapshots into its own locked output,
freezes an opponent/seed/seat plan and writes all attempted games. Comparisons use
the same plan and bootstrap whole seed blocks. Failed plans are not promoted to
the last valid comparison. The dashboard embeds only aggregate reports and can
be rebuilt without models or engine games.
