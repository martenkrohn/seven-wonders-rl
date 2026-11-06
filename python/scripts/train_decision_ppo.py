#!/usr/bin/env python3
"""Train a fresh BGA decision policy; legacy checkpoints are incompatible."""
import argparse,json
from pathlib import Path
from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.utils.decision_ppo import train,evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--agent-seat',type=int,choices=range(3),default=0)
    p.add_argument('--eval-games',type=int,default=3)
    p.add_argument('--resume',type=Path)
    p.add_argument('--allow-development',action='store_true')
    a=p.parse_args()
    env=DecisionEnv(a.engine,a.agent_seat,allow_development=a.allow_development)
    try:
        model,metrics=train(env,a.steps,a.seed,a.output_dir,a.resume)
        metrics['evaluation']=evaluate(env,model,a.eval_games,a.seed+1000)
        (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'steps':metrics['steps'],'completed_games':metrics['completed_games'],
                          'actually_applied':metrics['actually_applied'],'evaluation':metrics['evaluation']},indent=2))
    finally:env.close()
if __name__=='__main__':main()
