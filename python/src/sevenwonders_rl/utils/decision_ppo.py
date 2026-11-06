"""PPO for offered-option observations, with an incompatible explicit contract."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from sevenwonders_rl.features.decision_features import K,PUBLIC_DIM,CONTEXT_DIM,N_OPTIONS


def tensors(observations):
    return {key:torch.as_tensor(np.stack([o[key] for o in observations]),dtype=torch.float32)
            for key in observations[0]}


class DecisionActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.state=nn.Sequential(nn.Linear(3*PUBLIC_DIM+CONTEXT_DIM,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.candidate=nn.Sequential(nn.Linear(K,64),nn.Tanh())
        self.actor=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        self.critic=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))

    def forward(self,obs):
        mask=obs['action_mask'].bool()
        if not mask.any(dim=-1).all():raise ValueError('Policy received an observation without legal options')
        state=self.state(torch.cat((obs['public_table'].flatten(1),obs['context']),dim=-1))
        candidates=self.candidate(obs['action_feats'])
        logits=self.actor(torch.cat((state[:,None,:].expand(-1,N_OPTIONS,-1),candidates),dim=-1)).squeeze(-1)
        logits=logits.masked_fill(~mask,torch.finfo(logits.dtype).min)
        pooled=(candidates*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
        value=self.critic(torch.cat((state,pooled),dim=-1)).squeeze(-1)
        return logits,value

    @torch.no_grad()
    def select_action(self,obs,rng=None,deterministic=False):
        logits,_=self(tensors([obs]))
        if deterministic:return int(logits.argmax(-1).item())
        probabilities=logits.softmax(-1)[0].cpu().numpy().astype(np.float64)
        probabilities/=probabilities.sum()
        return int((rng or np.random.default_rng()).choice(N_OPTIONS,p=probabilities))


def save_checkpoint(path,model,optimizer,steps,contract,metrics):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data={'format':'decision-ppo-v1','contract':contract,'model':model.state_dict(),
          'optimizer':optimizer.state_dict(),'steps':steps,'metrics':metrics,'torch_rng':torch.get_rng_state()}
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(data,temporary);temporary.replace(path)


def load_checkpoint(path,contract,optimizer=None):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('format')!='decision-ppo-v1' or data.get('contract')!=contract:
        raise ValueError('Checkpoint decision/data/rules/feature contract mismatch; legacy models cannot be reused')
    model=DecisionActorCritic();model.load_state_dict(data['model'],strict=True)
    if optimizer is not None:raise ValueError('Create an optimizer for the returned model, then restore optimizer state')
    return model,data


def train(env,steps=128,seed=42,output=None,resume=None,rollout_size=64,epochs=4):
    if steps<1 or rollout_size<1 or epochs<1:raise ValueError('Positive training sizes required')
    torch.set_num_threads(1);torch.manual_seed(seed)
    obs,info=env.reset(seed=seed);contract=info['feature_contract']
    model=DecisionActorCritic();offset=0;checkpoint=None
    if resume:
        model,checkpoint=load_checkpoint(resume,contract);offset=checkpoint['steps']
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']);torch.set_rng_state(checkpoint['torch_rng'])
    episode=0;losses=[];applied=0;terminal=False
    for start in range(0,steps,rollout_size):
        observations=[];actions=[];old_logs=[];values=[];rewards=[];dones=[]
        size=min(rollout_size,steps-start)
        for n in range(size):
            if terminal:episode+=1;obs,info=env.reset(seed=seed+episode);terminal=False
            with torch.no_grad():
                logits,value=model(tensors([obs]));distribution=Categorical(logits=logits)
                action=distribution.sample();log=distribution.log_prob(action)
            observations.append(obs);actions.append(int(action));old_logs.append(float(log));values.append(float(value))
            obs,reward,terminal,truncated,info=env.step(int(action))
            if truncated:raise RuntimeError('Failed exchange cannot enter a PPO rollout')
            if info['transition']['receipt']['status']!='applied':raise RuntimeError('Unapplied rollout action')
            applied+=1;rewards.append(reward);dones.append(terminal)
        with torch.no_grad():bootstrap=0.0 if terminal else float(model(tensors([obs]))[1])
        advantages=[0.0]*size;gae=0.0;next_value=bootstrap
        for n in reversed(range(size)):
            alive=1.0-float(dones[n]);delta=rewards[n]+.99*alive*next_value-values[n]
            gae=delta+.99*.95*alive*gae;advantages[n]=gae;next_value=values[n]
        returns=torch.tensor(advantages)+torch.tensor(values)
        adv=torch.tensor(advantages);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        batch=tensors(observations);a=torch.tensor(actions);old=torch.tensor(old_logs)
        for _ in range(epochs):
            for indices in torch.randperm(size).split(min(16,size)):
                logits,value=model({k:v[indices] for k,v in batch.items()});distribution=Categorical(logits=logits)
                ratio=(distribution.log_prob(a[indices])-old[indices]).exp()
                policy=-torch.minimum(ratio*adv[indices],ratio.clamp(.8,1.2)*adv[indices]).mean()
                loss=policy+.5*(value-returns[indices]).square().mean()-.01*distribution.entropy().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),.5)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite PPO gradient')
                optimizer.step();losses.append(float(loss.detach()))
    metrics={'steps':offset+steps,'new_steps':steps,'actually_applied':applied,
             'completed_games':episode+int(terminal),'updates':len(losses),'loss_mean':float(np.mean(losses)),
             'contract':contract,'engine':info['engine'],'seed':seed,
             'resume_semantics':'Fresh engine game; model/optimizer/Torch state restored; no midgame engine restoration'}
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        save_checkpoint(output/'checkpoint.pt',model,optimizer,offset+steps,contract,metrics)
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    return model,metrics


def evaluate(env,model,games=3,seed=1000):
    results=[]
    for episode in range(games):
        obs,info=env.reset(seed=seed+episode)
        while True:
            action=model.select_action(obs,deterministic=True)
            obs,reward,done,truncated,info=env.step(action)
            if truncated:raise RuntimeError('Evaluation exchange failed')
            if done:results.append({'seed':seed+episode,'totals':info['final_totals'],'margin':reward});break
    return results
