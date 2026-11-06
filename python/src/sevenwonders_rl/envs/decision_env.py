"""Active BGA Gymnasium environment: one step per offered learner decision."""
from copy import deepcopy
from pathlib import Path
import json
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sevenwonders_engine import DecisionClient,EngineError
from sevenwonders_rl.features.decision_features import (N_OPTIONS,K,PUBLIC_DIM,CONTEXT_DIM,
    dataset,observation,feature_contract)
ROOT=Path(__file__).resolve().parents[4]


class DecisionEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,engine_path=None,agent_seat=0,opponent_policies=None,reward_scale=1.0,
                 allow_development=False,max_wait_seconds=10):
        super().__init__()
        if agent_seat not in range(3) or not np.isfinite(reward_scale) or reward_scale<=0:
            raise ValueError('Invalid learner seat/reward scale')
        self.explicit_engine=engine_path is not None
        default=ROOT/'engine/build/7Wonders'
        if not default.is_file():default=ROOT/'build/7Wonders'
        self.engine_path=Path(engine_path or default).resolve()
        self.agent_seat,self.opponents,self.reward_scale=agent_seat,opponent_policies or {},reward_scale
        self.allow_development,self.timeout=allow_development,max_wait_seconds
        self.client=None;self.done=True;self.state=None;self.options=[];self.history=[]
        self.action_space=spaces.Discrete(N_OPTIONS)
        self.observation_space=spaces.Dict({
            'public_table':spaces.Box(-np.inf,np.inf,(3,PUBLIC_DIM),np.float32),
            'context':spaces.Box(-np.inf,np.inf,(CONTEXT_DIM,),np.float32),
            'action_feats':spaces.Box(-np.inf,np.inf,(N_OPTIONS,K),np.float32),
            'action_mask':spaces.MultiBinary(N_OPTIONS)})

    def _selectable(self,seat):
        decision=next((d for d in self.state['decisions'] if d['actor']==seat),None)
        return [] if decision is None else [o for o in decision['options'] if o['selectable']]

    def _opponent_requests(self):
        requests=[]
        for decision in self.state['decisions']:
            seat=decision['actor']
            if seat==self.agent_seat:continue
            options=self._selectable(seat)
            obs=observation(self.state,seat,options)
            policy=self.opponents.get(seat)
            action=int(policy(obs,self.rngs[seat])) if policy is not None else int(self.rngs[seat].integers(len(options)))
            if action<0 or action>=len(options):raise ValueError('Opponent selected an unoffered action')
            requests.append(self.client.request(seat,options[action]['option_id']))
        return requests

    def _advance(self):
        for _ in range(100):
            self.state=self.client.state
            if self.state['game']['phase']=='aborted':raise EngineError('Game aborted: '+self.state.get('abort_reason','unknown'))
            if self.state['game']['phase']=='finished':
                self.done=True;self.options=[];return
            self.options=self._selectable(self.agent_seat)
            if self.options:return
            requests=self._opponent_requests()
            if not requests:raise EngineError('Nonterminal game has no offered actor')
            self.client.send(requests);self.client.wait()
        raise EngineError('Opponent-only advancement did not reach a learner or terminal decision')

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.close()
        engine_seed=int(seed) if seed is not None else int(self.np_random.integers(0,2**64,dtype=np.uint64))
        self.rngs=[np.random.default_rng(np.random.SeedSequence([engine_seed,s,1103])) for s in range(3)]
        self.history=[]
        try:
            self.client=DecisionClient(self.engine_path,engine_seed,self.timeout)
            self.state=self.client.wait()
            engine=self.state['engine']
            allowed={'bga-second-edition'}|({'bga-second-edition-development'} if self.allow_development else set())
            if engine['ruleset'] not in allowed:raise EngineError('A released BGA engine is required; development use must be explicit')
            if (engine['card_data']['source_sha256']!=dataset()['source_hash']
                or engine['wonder_data']['source_sha256']!=dataset()['source_hash']
                or not engine['card_data']['effects_complete'] or not engine['wonder_data']['effects_complete']):
                raise EngineError('Engine and packaged BGA data/capabilities differ')
            if not self.allow_development:
                if not engine.get('second_edition_implemented'):raise EngineError('BGA acceptance is incomplete')
                if engine.get('source_dirty'):raise EngineError('A clean released binary is required')
                lock_path=ROOT/'engine.lock.json'
                if not self.explicit_engine and lock_path.is_file():
                    lock=json.loads(lock_path.read_text())
                    if lock['protocol_version']!=3 or engine['source_commit']!=lock['commit'] or engine['release']!=lock['release']:
                        raise EngineError('Default engine binary differs from the consumer lock')
            self.contract=feature_contract(engine['ruleset']);self.engine_metadata=deepcopy(engine)
            self.done=False;self._advance()
            return observation(self.state,self.agent_seat,self.options),self._info()
        except Exception:
            self.close();raise

    def _info(self):
        # Deliberately no raw transport state or opponent private hands/options.
        info={'game_id':self.state['game_id'],'turn':self.state['game']['turn'],
              'phase':self.state['game']['phase'],'engine':deepcopy(self.engine_metadata),
              'feature_contract':deepcopy(self.contract),'decisions_applied':len(self.history)}
        if self.done:info['final_totals']=[self.state['result']['totals'][str(s)] for s in range(3)]
        return info

    def _verify_result(self):
        path=self.client.root/'results.txt';deadline=time.monotonic()+self.timeout
        while not path.exists():
            if time.monotonic()>deadline or self.client.process.poll() is not None:raise EngineError('Missing normal result file')
            time.sleep(.002)
        totals=self.state['result']['totals']
        expected=''.join(f'Player {s+1} score: {totals[str(s)]}\n' for s in range(3))
        if path.read_text()!=expected:raise EngineError('Terminal totals/result file differ')

    def step(self,action):
        if self.done or self.client is None:raise RuntimeError('reset() is required before step()')
        if not self.action_space.contains(action) or int(action)>=len(self.options):
            raise ValueError('Action is outside the current offered mask')
        option=deepcopy(self.options[int(action)]);before=deepcopy(self.state['game'])
        request=self.client.request(self.agent_seat,option['option_id'])
        requests=[request,*self._opponent_requests()]
        self.client.send(requests);self.client.wait();self._advance()
        receipt=next((r for r in self.state['receipts'] if r.get('request_id')==request['request_id']),None)
        if receipt is None or receipt['status']!='applied':raise EngineError('Learner action was not actually applied')
        transition={'decision_id':request['decision_id'],'offered_revision':request['offered_revision'],
            'turn_before':before['turn'],'turn_after':self.state['game']['turn'],
            'option':option,'receipt':deepcopy(receipt)}
        self.history.append(deepcopy(transition))
        reward=0.0
        if self.done:
            self._verify_result();totals=self.state['result']['totals']
            reward=(totals[str(self.agent_seat)]-max(v for s,v in totals.items() if int(s)!=self.agent_seat))*self.reward_scale
        info=self._info();info['transition']=transition
        return observation(self.state,self.agent_seat,self.options),float(reward),self.done,False,info

    def close(self):
        if self.client is not None:self.client.close();self.client=None
        self.done=True
