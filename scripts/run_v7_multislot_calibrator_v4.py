import os, sys, json, csv, math
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
from partcat_hkg.abg_aog_v7.diagnostics import record_label

FEATURES = [
    'slot_presence', 'slot_absence', 'slot_token', 'slot_geom', 'missing',
    'weak_missing', 'extra_unassigned', 'role', 'active_terms', 'matched_slots',
    'miss_count', 'multiplicity_gap'
]


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


def box_from_geom4(g):
    cx, cy, w, h = [float(x) for x in g]
    return (cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h)


def safe_lr(a,b):
    return float(torch.log(torch.tensor((float(a)+1e-4)/(float(b)+1e-4))).clamp(-2.5,2.5))


def norm_token(t):
    if t is None or not torch.is_tensor(t) or t.numel() == 0: return None
    return F.normalize(t.detach().float().flatten().cpu(), dim=0)


def choose_num_slots(per_image_counts, total_obs, num_images, max_slots):
    if num_images <= 0 or total_obs <= 0: return 0
    counts = sorted(int(c) for c in per_image_counts if int(c) > 0)
    if not counts: return 0
    q80 = counts[min(len(counts)-1, int(0.80*(len(counts)-1)))]
    mean = total_obs / max(1, num_images)
    k = max(1, int(round(max(mean, q80))))
    return int(max(1, min(max_slots, k)))


def simple_kmeans(X, k, iters=20):
    if X.shape[0] <= k:
        return X.clone(), torch.arange(X.shape[0])
    # deterministic farthest-point initialization on cx/cy/scale space
    centers = [int(torch.argmin(X[:,0] + X[:,1]).item())]
    while len(centers) < k:
        C = X[centers]
        d = torch.cdist(X, C).min(1).values
        centers.append(int(torch.argmax(d).item()))
    C = X[centers].clone()
    labels = torch.zeros(X.shape[0], dtype=torch.long)
    for _ in range(iters):
        labels = torch.cdist(X, C).argmin(1)
        new = []
        for j in range(k):
            mask = labels == j
            new.append(X[mask].mean(0) if bool(mask.any()) else C[j])
        newC = torch.stack(new)
        if torch.allclose(newC, C, atol=1e-5): break
        C = newC
    return C, labels


def role_score(class_name, part_names, terms):
    cn = class_name.lower()
    names = [part_names[t.functional_part_id].lower() if 0 <= int(t.functional_part_id) < len(part_names) else str(t.functional_part_id) for t in terms]
    def count(words): return sum(1 for n in names if any(w in n for w in words))
    limb=count(('leg','foot','paw','arm')); wheel=count(('wheel',)); wing=count(('wing',)); body=count(('body','torso','frame'))
    head=count(('head',)); tail=count(('tail',)); s=0.0
    if 'bicycle' in cn or 'bike' in cn: s += 0.8*min(wheel,2) - 0.8*max(0,2-wheel) + 0.25*body
    if 'snake' in cn: s += 0.35*body + 0.25*head + 0.25*tail - 0.45*limb - 0.60*wheel - 0.40*wing
    if 'quadruped' in cn: s += 0.30*body + 0.25*head + 0.35*min(limb,4) - 0.35*wheel
    if 'reptile' in cn: s += 0.30*body + 0.25*head + 0.25*tail + 0.10*min(limb,4) - 0.40*wing - 0.40*wheel
    if 'bird' in cn: s += 0.45*wing + 0.20*head + 0.10*body - 0.25*wheel
    if 'fish' in cn: s += 0.35*body + 0.20*tail - 0.35*limb - 0.35*wheel
    return float(s)


class SlotBank:
    def __init__(self, class_names, part_names, slots, global_part_rate, cfg):
        self.class_names = list(class_names); self.part_names = list(part_names)
        self.slots = slots; self.global_part_rate = global_part_rate; self.cfg = cfg
        self.by_class = defaultdict(list)
        for s in slots: self.by_class[int(s['class_id'])].append(s)
        self.class_ids = sorted(self.by_class.keys())

    def save_json(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'class_names':self.class_names,'part_names':self.part_names,'global_part_rate':self.global_part_rate,'cfg':self.cfg,'slots':self.slots}, indent=2), encoding='utf-8')

    @classmethod
    def load_json(cls, path):
        p = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(p['class_names'], p['part_names'], p['slots'], {int(k):float(v) for k,v in p['global_part_rate'].items()}, p.get('cfg',{}))


def build_slot_bank(records, class_names, part_names, score_tau=0.05, max_slots_per_part=6, required_tau=0.35, min_slot_support=3):
    obs_by_cp = defaultdict(list); counts_by_cp_img = defaultdict(Counter); class_counts = Counter(); global_part_img = Counter()
    for sid, rec in enumerate(records):
        y = record_label(rec)
        if y < 0: continue
        class_counts[y] += 1
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True)
        parts_seen = set()
        for t in terms:
            p = int(t.functional_part_id); parts_seen.add(p)
            tok = norm_token(t.appearance_token)
            obs_by_cp[(y,p)].append({'sample_id':sid, 'score':float(t.visible_score), 'geom':geom4(t.visible_box_xyxy), 'token':tok})
            counts_by_cp_img[(y,p)][sid] += 1
        global_part_img.update(parts_seen)
    n_all = max(1, len([r for r in records if record_label(r) >= 0]))
    global_rate = {int(p):(float(c)+1.0)/(n_all+2.0) for p,c in global_part_img.items()}
    slots = []
    for (c,p), obs in sorted(obs_by_cp.items()):
        n_cls = max(1, class_counts[c]); per_img = list(counts_by_cp_img[(c,p)].values())
        k = choose_num_slots(per_img, len(obs), n_cls, max_slots_per_part)
        if k <= 0: continue
        X = torch.stack([o['geom'] for o in obs])
        C, labels = simple_kmeans(X, k)
        for j in range(k):
            idx = torch.nonzero(labels == j, as_tuple=False).flatten().tolist()
            if len(idx) < min_slot_support and k > 1: continue
            rows = [obs[i] for i in idx]
            support_images = len(set(r['sample_id'] for r in rows))
            rate = (support_images + 1.0) / (n_cls + 2.0)
            gr = float(global_rate.get(p, 1.0 / max(2, len(part_names))))
            req = max(0.0, min(1.0, (rate - required_tau) / max(1e-6, 1.0-required_tau))) if rate >= required_tau else 0.0
            diag = safe_lr(rate, gr)
            G = torch.stack([r['geom'] for r in rows]); mean = G.mean(0); var = G.var(0, unbiased=False).clamp_min(0.01)
            toks = [r['token'] for r in rows if r['token'] is not None]
            token_mean = []
            if toks:
                token_mean = [float(x) for x in F.normalize(torch.stack(toks).mean(0), dim=0)]
            slots.append({'class_id':int(c),'class_name':class_names[c] if 0 <= c < len(class_names) else f'class_{c}', 'part_id':int(p), 'part_name':part_names[p] if 0 <= p < len(part_names) else f'part_{p}', 'slot_id':int(j), 'support':int(len(rows)), 'support_images':int(support_images), 'rate':float(rate), 'global_rate':float(gr), 'diagnostic':float(diag), 'requiredness':float(req), 'geom_mean':[float(x) for x in mean], 'geom_var':[float(x) for x in var], 'token_mean':token_mean, 'token_support':len(toks)})
    return SlotBank(class_names, part_names, slots, global_rate, {'score_tau':score_tau,'max_slots_per_part':max_slots_per_part,'required_tau':required_tau,'min_slot_support':min_slot_support})


def match_slots_for_class(bank, terms, cid, score_tau=0.05):
    slots = bank.by_class.get(int(cid), [])
    terms_by_part = defaultdict(list)
    for i,t in enumerate(terms):
        if t.visible_score >= score_tau: terms_by_part[int(t.functional_part_id)].append((i,t))
    matches = {}; used_terms = set()
    # greedy by requiredness/support first
    for si, s in sorted(enumerate(slots), key=lambda x:(-x[1]['requiredness'], -x[1]['rate'], x[1]['part_id'], x[1]['slot_id'])):
        p = int(s['part_id']); best = None; best_score = -1e9
        mu = torch.tensor(s['geom_mean'], dtype=torch.float32); var = torch.tensor(s['geom_var'], dtype=torch.float32).clamp_min(1e-3)
        for ti,t in terms_by_part.get(p, []):
            if ti in used_terms: continue
            g = geom4(t.visible_box_xyxy)
            geom_sim = float(torch.exp(-0.5*torch.clamp((((g-mu)**2)/var).mean(), max=4.0))) if mu.numel() == g.numel() else 0.0
            tok_sim = 0.0
            if s.get('token_mean') and t.appearance_token is not None:
                obs = F.normalize(t.appearance_token.float().flatten(), dim=0); proto=torch.tensor(s['token_mean']).float()
                if obs.numel() == proto.numel(): tok_sim = float(torch.dot(obs, F.normalize(proto, dim=0)).clamp(-1,1))
            sc = float(t.visible_score) * (0.65*geom_sim + 0.35*max(tok_sim,0.0))
            if sc > best_score: best_score = sc; best = (ti,t,geom_sim,tok_sim)
        if best is not None:
            ti,t,gs,ts = best; used_terms.add(ti); matches[si] = {'term_idx':ti,'term':t,'geom_sim':gs,'token_sim':ts,'match_score':best_score}
    return slots, matches, used_terms


def candidate_features(bank, terms, cid, score_tau=0.05):
    slots, matches, used = match_slots_for_class(bank, terms, cid, score_tau=score_tau)
    f = {k:0.0 for k in FEATURES}; f['active_terms'] = float(sum(1 for t in terms if t.visible_score >= score_tau))
    for si,s in enumerate(slots):
        m = matches.get(si); obs = 0.0 if m is None else float(m['term'].visible_score)
        f['slot_absence'] += (1.0-obs) * safe_lr(1.0-s['rate'], 1.0-s['global_rate']) * max(0.25, float(s['requiredness']))
        if m is None:
            if float(s['requiredness']) > 0:
                f['missing'] += float(s['requiredness']); f['miss_count'] += 1.0
            continue
        f['matched_slots'] += 1.0
        f['slot_presence'] += obs * float(s['diagnostic'])
        f['slot_geom'] += obs * float(m['geom_sim'])
        f['slot_token'] += obs * float(m['token_sim'])
        if float(s['requiredness']) > 0 and obs < 0.35:
            f['weak_missing'] += float(s['requiredness']) * (0.35 - obs)
    f['extra_unassigned'] = float(sum(t.visible_score for i,t in enumerate(terms) if t.visible_score >= score_tau and i not in used))
    expected = sum(1.0 for s in slots if float(s['requiredness']) > 0)
    f['multiplicity_gap'] = abs(expected - f['matched_slots'])
    f['role'] = role_score(bank.class_names[cid] if 0 <= cid < len(bank.class_names) else str(cid), bank.part_names, terms)
    return f


def make_tensor(bank, records, score_tau=0.05):
    cids = sorted(bank.by_class.keys()); X=[]; y=[]; cache=[]
    for sid,rec in enumerate(records):
        yy = record_label(rec); terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True)
        if yy not in cids: continue
        row=[]; row_cache=[]
        for c in cids:
            fd = candidate_features(bank, terms, c, score_tau=score_tau)
            row.append([fd[k] for k in FEATURES]); row_cache.append((sid,yy,c,fd))
        X.append(row); y.append(cids.index(yy)); cache.append(row_cache)
    return cids, torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), cache


def train_calibrator(X, y, epochs=800, lr=0.05, wd=1e-3, seed=7):
    torch.manual_seed(seed)
    mu = X.flatten(0,1).mean(0); sig = X.flatten(0,1).std(0).clamp_min(1e-4)
    Z = (X-mu)/sig
    w = torch.zeros(Z.shape[-1], requires_grad=True); b = torch.zeros(Z.shape[1], requires_grad=True)
    opt = torch.optim.AdamW([w,b], lr=lr, weight_decay=wd); logs=[]
    for ep in range(int(epochs)):
        logits = torch.einsum('ncf,f->nc', Z, w) + b
        loss = F.cross_entropy(logits, y) + 1e-3*(w*w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs-1:
            logs.append({'epoch':ep, 'loss':float(loss.detach()), 'acc':float((logits.argmax(1)==y).float().mean())})
    return {'w':w.detach(), 'b':b.detach(), 'mu':mu, 'sig':sig, 'logs':logs}


def eval_and_write(out_dir, bank, cids, X, y, cache, model):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    logits = torch.einsum('ncf,f->nc', (X-model['mu'])/model['sig'], model['w']) + model['b']
    pred_idx = logits.argmax(1); correct = (pred_idx == y)
    all_rows=[]; top_rows=[]; flags=[]; conf=Counter()
    for i,row_cache in enumerate(cache):
        yy = int(row_cache[0][1]); pred_class = int(cids[int(pred_idx[i])]); conf[(yy,pred_class)] += 1
        order = torch.argsort(logits[i], descending=True).tolist(); true_rank = order.index(int(y[i]))+1 if int(y[i]) in order else -1
        for rank,j in enumerate(order,1):
            sid,y0,c,fd = row_cache[j]
            row = {'sample_id':sid,'true_class':y0,'pred_class':pred_class,'candidate_class':c,'candidate_name':bank.class_names[c] if 0 <= c < len(bank.class_names) else str(c),'rank':rank,'calibrated_score':float(logits[i,j])}
            row.update(fd)
            for k,name in enumerate(FEATURES): row['w_'+name] = float(((X[i,j,k]-model['mu'][k])/model['sig'][k])*model['w'][k])
            all_rows.append(row)
            if rank <= 11: top_rows.append(row)
        winner = row_cache[int(pred_idx[i])][3]
        flags.append({'sample_id':row_cache[0][0], 'true_class':yy, 'pred_class':pred_class, 'correct':bool(correct[i]), 'true_class_rank':true_rank, 'true_class_not_in_top5':true_rank<0 or true_rank>5, 'missing_required_pred':int(winner['miss_count'])})
    write_csv(out/'class_score_decomposition.csv', all_rows); write_csv(out/'candidate_scores_topn.csv', top_rows); write_csv(out/'failure_flags_by_sample.csv', flags)
    write_csv(out/'confusion_matrix_long.csv', [{'true_class':a,'pred_class':b,'count':c} for (a,b),c in sorted(conf.items())])
    summary = {'samples':len(flags), 'accuracy':float(correct.float().mean()), 'pred_distribution':dict(Counter(f['pred_class'] for f in flags)), 'true_class_not_in_top5':sum(f['true_class_not_in_top5'] for f in flags), 'missing_required_pred':sum(f['missing_required_pred']>0 for f in flags), 'num_slots':len(bank.slots), 'weights':{name:float(model['w'][i]) for i,name in enumerate(FEATURES)}, 'bias':{str(cids[i]):float(model['b'][i]) for i in range(len(cids))}, 'train_logs':model.get('logs',[])}
    (out/'diagnostic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    bank.save_json(out/'multislot_bank.json'); torch.save(model, out/'calibrator.pt')
    print(json.dumps(summary, indent=2))
    return summary


def visualize_layouts(bank, out_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except Exception:
        return
    cids = sorted(bank.by_class.keys()); n=len(cids); cols=4; rows=math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4)); axes = axes.flatten() if hasattr(axes,'flatten') else [axes]
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    for ax,c in zip(axes,cids):
        ax.set_xlim(0,1); ax.set_ylim(1,0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ss = sorted(bank.by_class[c], key=lambda s:(s['part_id'], s['slot_id']))
        ax.set_title(f"{bank.class_names[c] if 0<=c<len(bank.class_names) else c}\nslots={len(ss)}")
        for s in ss:
            box = box_from_geom4(s['geom_mean']); x0,y0,x1,y1=box
            color = colors[int(s['part_id']) % len(colors)]
            alpha = 0.18 + 0.45*min(1.0, float(s['requiredness']))
            ax.add_patch(patches.Rectangle((x0,y0),x1-x0,y1-y0,fill=True,alpha=alpha,edgecolor=color,facecolor=color,linewidth=2))
            ax.text((x0+x1)/2,(y0+y1)/2,f"{s['part_name']}:{s['slot_id']}",ha='center',va='center',fontsize=7)
    for ax in axes[n:]: ax.axis('off')
    fig.suptitle('Multi-slot class templates: functional part categories can have multiple slots')
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, dpi=180); plt.close(fig)


def main():
    train_cache=os.environ.get('TRAIN_CACHE'); val_cache=os.environ.get('VAL_CACHE')
    if not train_cache or not val_cache: raise SystemExit('Set TRAIN_CACHE and VAL_CACHE')
    score_tau=float(os.environ.get('SCORE_TAU','0.05')); out=Path(os.environ.get('OUT_DIR','runs/v7_multislot_calibrator_v4'))
    max_slots=int(os.environ.get('MAX_SLOTS_PER_PART','6'))
    train=load_terminal_cache(train_cache, map_location='cpu', materialize=True); val=load_terminal_cache(val_cache, map_location='cpu', materialize=True)
    schema=RoleSchema.from_payload(train['schema'])
    bank=build_slot_bank(list(train.get('records',[])), list(schema.class_names), list(schema.part_names), score_tau=score_tau, max_slots_per_part=max_slots, required_tau=float(os.environ.get('REQUIRED_TAU','0.35')), min_slot_support=int(os.environ.get('MIN_SLOT_SUPPORT','3')))
    visualize_layouts(bank, out/'multislot_template_layouts.png')
    cids,Xtr,ytr,_=make_tensor(bank, list(train.get('records',[])), score_tau=score_tau)
    model=train_calibrator(Xtr,ytr,epochs=int(os.environ.get('EPOCHS','800')),lr=float(os.environ.get('LR','0.05')),wd=float(os.environ.get('WD','0.001')))
    cids,Xv,yv,cache=make_tensor(bank, list(val.get('records',[])), score_tau=score_tau)
    eval_and_write(out, bank, cids, Xv, yv, cache, model)


if __name__ == '__main__':
    main()
