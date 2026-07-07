import os, sys, json, csv
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.diagnostics import build_profile_bank_v7, record_label


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def geom4(box):
    x0, y0, x1, y1 = [float(x) for x in box]
    return torch.tensor([(x0+x1)/2, (y0+y1)/2, max(x1-x0,1e-4), max(y1-y0,1e-4)], dtype=torch.float32)


def safe_lr(a,b):
    return float(torch.log(torch.tensor((float(a)+1e-4)/(float(b)+1e-4))).clamp(-2.5,2.5))


def add_token_prototypes(bank, records, score_tau=0.05):
    buckets = defaultdict(list)
    for rec in records:
        y = record_label(rec)
        if y < 0: continue
        best = {}
        for t in terminal_packets_from_record(rec, score_tau=score_tau, include_tokens=True):
            if t.appearance_token is None: continue
            p = int(t.functional_part_id)
            if p not in best or t.visible_score > best[p].visible_score:
                best[p] = t
        for p, t in best.items():
            buckets[(y,p)].append(F.normalize(t.appearance_token.float().flatten(), dim=0).cpu())
    for (c,p), xs in buckets.items():
        prof = bank.profiles.get(c)
        if prof is not None and p in prof.parts:
            prof.parts[p].token_mean = F.normalize(torch.stack(xs).mean(0), dim=0)
            prof.parts[p].token_support = len(xs)
    return bank


def part_name(part_names, pid):
    return part_names[pid].lower() if 0 <= int(pid) < len(part_names) else str(pid)


def role_score(class_name, part_names, terms):
    cn = class_name.lower()
    names = [part_name(part_names, t.functional_part_id) for t in terms]
    def count(words): return sum(1 for n in names if any(w in n for w in words))
    limb = count(('leg','foot','paw','arm'))
    wheel = count(('wheel',)); wing = count(('wing',)); body = count(('body','torso','frame'))
    head = count(('head',)); tail = count(('tail',)); s = 0.0
    if 'bicycle' in cn or 'bike' in cn: s += 0.8*min(wheel,2) - 0.8*max(0,2-wheel) + 0.25*body
    if 'snake' in cn: s += 0.35*body + 0.25*head + 0.25*tail - 0.45*limb - 0.60*wheel - 0.40*wing
    if 'quadruped' in cn: s += 0.30*body + 0.25*head + 0.35*min(limb,2) - 0.35*wheel
    if 'reptile' in cn: s += 0.30*body + 0.25*head + 0.25*tail + 0.10*min(limb,2) - 0.40*wing - 0.40*wheel
    if 'bird' in cn: s += 0.45*wing + 0.20*head + 0.10*body - 0.25*wheel
    if 'fish' in cn: s += 0.35*body + 0.20*tail - 0.35*limb - 0.35*wheel
    return float(s)


FEATS = ['presence','absence','token','geom','missing','weak_missing','extra','role','active','covered','miss_count']


def candidate_features(bank, terms, cid, score_tau=0.05):
    prof = bank.profiles[cid]
    best = {}
    for t in terms:
        if t.visible_score < score_tau: continue
        p = int(t.functional_part_id)
        if p not in best or t.visible_score > best[p].visible_score: best[p] = t
    f = {k:0.0 for k in FEATS}; f['active'] = float(len(best))
    covered = 0
    for p,t in best.items():
        st = prof.parts.get(p)
        if st is None:
            f['extra'] += float(t.visible_score); continue
        covered += 1
        f['presence'] += float(t.visible_score) * float(st.diagnostic)
        tm = getattr(st, 'token_mean', None)
        if tm is not None and t.appearance_token is not None:
            obs = F.normalize(t.appearance_token.float().flatten(), dim=0); proto = torch.as_tensor(tm).float()
            if obs.numel() == proto.numel():
                f['token'] += float(t.visible_score) * float(torch.dot(obs, F.normalize(proto, dim=0)).clamp(-1,1))
        mu = torch.tensor(st.geom_mean, dtype=torch.float32); vv = torch.tensor(st.geom_var, dtype=torch.float32).clamp_min(1e-3); ob = geom4(t.visible_box_xyxy)
        if mu.numel() == ob.numel():
            f['geom'] += float(t.visible_score) * float(torch.exp(-0.5*torch.clamp((((ob-mu)**2)/vv).mean(), max=4.0)))
    f['covered'] = float(covered)
    for p, st in prof.parts.items():
        obs = float(best[p].visible_score) if p in best else 0.0
        req = float(st.requiredness)
        # Positive means absent evidence supports this class; negative means it hurts.
        f['absence'] += (1.0 - obs) * safe_lr(1.0-float(st.rate), 1.0-float(st.global_rate)) * max(0.25, req)
        if req > 0 and obs < score_tau:
            f['missing'] += req * (1.0 - obs); f['miss_count'] += 1.0
        elif req > 0 and obs < 0.35:
            f['weak_missing'] += req * (0.35 - obs)
    f['role'] = role_score(prof.class_name, bank.part_names, terms)
    return f


def make_tensor(bank, records, score_tau=0.05):
    cids = sorted(bank.profiles.keys()); X = []; y = []
    rows_cache = []
    for sid, rec in enumerate(records):
        yy = record_label(rec); terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True)
        sample = []; cache = []
        for c in cids:
            fd = candidate_features(bank, terms, c, score_tau=score_tau)
            sample.append([fd[k] for k in FEATS]); cache.append((sid, yy, c, fd))
        X.append(sample); y.append(cids.index(yy) if yy in cids else -1); rows_cache.append(cache)
    keep = [i for i,v in enumerate(y) if v >= 0]
    X = torch.tensor([X[i] for i in keep], dtype=torch.float32)
    y = torch.tensor([y[i] for i in keep], dtype=torch.long)
    cache = [rows_cache[i] for i in keep]
    return cids, X, y, cache


def train_calibrator(X, y, epochs=800, lr=0.05, wd=1e-3, seed=3):
    torch.manual_seed(seed)
    mu = X.flatten(0,1).mean(0); sig = X.flatten(0,1).std(0).clamp_min(1e-4)
    Z = (X - mu) / sig
    w = torch.zeros(Z.shape[-1], requires_grad=True); b = torch.zeros(Z.shape[1], requires_grad=True)
    opt = torch.optim.AdamW([w,b], lr=lr, weight_decay=wd)
    logs=[]
    for ep in range(int(epochs)):
        logits = torch.einsum('ncf,f->nc', Z, w) + b
        loss = F.cross_entropy(logits, y) + 1e-3*(w*w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs-1:
            acc = float((logits.argmax(1)==y).float().mean())
            logs.append({'epoch':ep,'loss':float(loss.detach()),'acc':acc})
    return {'w':w.detach(), 'b':b.detach(), 'mu':mu, 'sig':sig, 'logs':logs}


def eval_and_write(out_dir, cids, X, y, cache, model):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    Z = (X - model['mu']) / model['sig']
    logits = torch.einsum('ncf,f->nc', Z, model['w']) + model['b']
    pred_idx = logits.argmax(1); correct = (pred_idx == y)
    all_rows=[]; top_rows=[]; flags=[]; conf=Counter()
    for i, sample_cache in enumerate(cache):
        yy = int(sample_cache[0][1]); pred_class = int(cids[int(pred_idx[i])]); conf[(yy,pred_class)] += 1
        order = torch.argsort(logits[i], descending=True).tolist(); true_rank = order.index(int(y[i]))+1 if int(y[i]) in order else -1
        for rank, j in enumerate(order, 1):
            sid, y0, c, fd = sample_cache[j]; row = {'sample_id':sid,'true_class':y0,'pred_class':pred_class,'candidate_class':c,'rank':rank,'calibrated_score':float(logits[i,j])}
            row.update(fd)
            # weighted contribution after standardization, for debugging.
            for k, name in enumerate(FEATS): row['w_'+name] = float(((X[i,j,k]-model['mu'][k])/model['sig'][k])*model['w'][k])
            all_rows.append(row)
            if rank <= 11: top_rows.append(row)
        winner = sample_cache[int(pred_idx[i])][3]
        flags.append({'sample_id':sample_cache[0][0], 'true_class':yy, 'pred_class':pred_class, 'correct':bool(correct[i]), 'true_class_rank':true_rank, 'true_class_not_in_top5':true_rank<0 or true_rank>5, 'missing_required_pred':int(winner['miss_count'])})
    write_csv(out/'class_score_decomposition.csv', all_rows); write_csv(out/'candidate_scores_topn.csv', top_rows); write_csv(out/'failure_flags_by_sample.csv', flags)
    write_csv(out/'confusion_matrix_long.csv', [{'true_class':a,'pred_class':b,'count':c} for (a,b),c in sorted(conf.items())])
    summary = {'samples':len(flags), 'accuracy':float(correct.float().mean()), 'pred_distribution':dict(Counter(f['pred_class'] for f in flags)), 'true_class_not_in_top5':sum(f['true_class_not_in_top5'] for f in flags), 'missing_required_pred':sum(f['missing_required_pred']>0 for f in flags), 'weights':{name:float(model['w'][i]) for i,name in enumerate(FEATS)}, 'bias':{str(cids[i]):float(model['b'][i]) for i in range(len(cids))}, 'train_logs':model.get('logs',[])}
    (out/'diagnostic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    torch.save(model, out/'calibrator.pt')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    train_cache = os.environ.get('TRAIN_CACHE'); val_cache = os.environ.get('VAL_CACHE')
    if not train_cache or not val_cache: raise SystemExit('Set TRAIN_CACHE and VAL_CACHE')
    score_tau = float(os.environ.get('SCORE_TAU','0.05')); out = Path(os.environ.get('OUT_DIR','runs/v7_learned_calibrator_v3'))
    train = load_terminal_cache(train_cache, map_location='cpu', materialize=True); val = load_terminal_cache(val_cache, map_location='cpu', materialize=True)
    schema = RoleSchema.from_payload(train['schema'])
    bank = build_profile_bank_v7(list(train.get('records',[])), class_names=list(schema.class_names), part_names=list(schema.part_names), score_tau=score_tau, required_tau=0.35, min_relation_support=6, uniform_prior=True)
    bank = add_token_prototypes(bank, list(train.get('records',[])), score_tau=score_tau)
    cids, Xtr, ytr, _ = make_tensor(bank, list(train.get('records',[])), score_tau=score_tau)
    model = train_calibrator(Xtr, ytr, epochs=int(os.environ.get('EPOCHS','800')), lr=float(os.environ.get('LR','0.05')), wd=float(os.environ.get('WD','0.001')))
    cids, Xv, yv, cache = make_tensor(bank, list(val.get('records',[])), score_tau=score_tau)
    eval_and_write(out, cids, Xv, yv, cache, model)


if __name__ == '__main__': main()
