import os, sys, json
from pathlib import Path
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.diagnostics import build_profile_bank_v7, write_csv, record_label


def add_token_prototypes(bank, records, score_tau=0.05):
    toks = defaultdict(list)
    for rec in records:
        y = record_label(rec)
        if y < 0: continue
        best = {}
        for t in terminal_packets_from_record(rec, score_tau=score_tau, include_tokens=True):
            if t.appearance_token is None: continue
            p = t.functional_part_id
            if p not in best or t.visible_score > best[p].visible_score: best[p] = t
        for p, t in best.items():
            toks[(y, p)].append(F.normalize(t.appearance_token.float().flatten(), dim=0).cpu())
    for (c, p), xs in toks.items():
        prof = bank.profiles.get(c)
        if prof and p in prof.parts:
            prof.parts[p].token_mean = F.normalize(torch.stack(xs).mean(0), dim=0)
            prof.parts[p].token_support = len(xs)
    return bank


def geom4(box):
    x0,y0,x1,y1=[float(x) for x in box]
    return torch.tensor([(x0+x1)/2,(y0+y1)/2,max(x1-x0,1e-4),max(y1-y0,1e-4)])


def lratio(a,b):
    return float(torch.log(torch.tensor((float(a)+1e-4)/(float(b)+1e-4))).clamp(-2.5,2.5))


def score(bank, terms, sid, score_tau=0.05, token_w=0.45, miss_w=1.4, abs_w=0.75):
    best={}
    for t in terms:
        if t.visible_score < score_tau: continue
        p=t.functional_part_id
        if p not in best or t.visible_score > best[p].visible_score: best[p]=t
    rows=[]
    for c, prof in bank.profiles.items():
        r={'sample_id':sid,'candidate_class':c,'candidate_name':prof.class_name,'node_presence_score':0.0,'node_absence_score':0.0,'token_prototype_score':0.0,'part_template_score':0.0,'required_slot_penalty':0.0,'extra_part_penalty':0.0,'total_score':0.0,'missing_required':0,'rank':-1}
        for p,t in best.items():
            st=prof.parts.get(p)
            if st is None:
                r['extra_part_penalty']-=0.08*float(t.visible_score); continue
            r['node_presence_score']+=float(t.visible_score)*float(st.diagnostic)
            tm=getattr(st,'token_mean',None)
            if tm is not None and t.appearance_token is not None:
                obs=F.normalize(t.appearance_token.float().flatten(),dim=0); proto=torch.as_tensor(tm).float()
                if obs.numel()==proto.numel(): r['token_prototype_score']+=token_w*float(t.visible_score)*float(torch.dot(obs,F.normalize(proto,dim=0)).clamp(-1,1))
            mu=torch.tensor(st.geom_mean); vv=torch.tensor(st.geom_var).clamp_min(1e-3); ob=geom4(t.visible_box_xyxy)
            if mu.numel()==ob.numel(): r['part_template_score']+=0.25*float(t.visible_score)*float(torch.exp(-0.5*torch.clamp((((ob-mu)**2)/vv).mean(),max=4.0)))
        for p,st in prof.parts.items():
            obs=float(best[p].visible_score) if p in best else 0.0
            r['node_absence_score']+=abs_w*(1.0-obs)*lratio(1.0-st.rate,1.0-st.global_rate)*max(0.25,float(st.requiredness))
            if float(st.requiredness)>0 and obs<score_tau:
                r['missing_required']+=1; r['required_slot_penalty']-=miss_w*float(st.requiredness)*(1.0-obs)
        r['total_score']=r['node_presence_score']+r['node_absence_score']+r['token_prototype_score']+r['part_template_score']+r['required_slot_penalty']+r['extra_part_penalty']
        rows.append(r)
    rows.sort(key=lambda x:x['total_score'], reverse=True)
    for i,r in enumerate(rows,1): r['rank']=i
    return rows


def main():
    train=load_terminal_cache(os.environ['TRAIN_CACHE'], map_location='cpu', materialize=True)
    val=load_terminal_cache(os.environ['VAL_CACHE'], map_location='cpu', materialize=True)
    schema=RoleSchema.from_payload(train['schema'])
    class_names=list(getattr(schema,'class_names',getattr(schema,'obj_names',[])))
    bank=build_profile_bank_v7(list(train.get('records',[])), class_names=class_names, part_names=list(schema.part_names), score_tau=0.05, required_tau=0.35, min_relation_support=6, uniform_prior=True)
    bank=add_token_prototypes(bank, list(train.get('records',[])))
    out=Path(os.environ.get('OUT_DIR','runs/v7_balanced_v2')); out.mkdir(parents=True, exist_ok=True)
    all_rows=[]; top=[]; flags=[]; correct=0; conf=Counter()
    for sid,rec in enumerate(val.get('records',[])):
        y=record_label(rec); terms=terminal_packets_from_record(rec, sample_id=sid, score_tau=0.05, include_tokens=True); rows=score(bank,terms,sid)
        pred=rows[0]['candidate_class'] if rows else -1; correct+=int(pred==y); conf[(y,pred)]+=1; tr=next((r['rank'] for r in rows if r['candidate_class']==y),-1)
        for r in rows: z=dict(r); z['true_class']=y; z['pred_class']=pred; all_rows.append(z)
        for r in rows[:11]: z=dict(r); z['true_class']=y; z['pred_class']=pred; top.append(z)
        flags.append({'sample_id':sid,'true_class':y,'pred_class':pred,'correct':pred==y,'true_class_rank':tr,'true_class_not_in_top5':tr<0 or tr>5,'missing_required_pred':rows[0]['missing_required'] if rows else 0})
    write_csv(out/'class_score_decomposition.csv', all_rows); write_csv(out/'candidate_scores_topn.csv', top); write_csv(out/'failure_flags_by_sample.csv', flags); write_csv(out/'confusion_matrix_long.csv', [{'true_class':a,'pred_class':b,'count':c} for (a,b),c in sorted(conf.items())])
    s={'samples':len(flags),'accuracy':correct/max(1,len(flags)),'pred_distribution':dict(Counter(f['pred_class'] for f in flags)),'true_class_not_in_top5':sum(1 for f in flags if f['true_class_not_in_top5']),'missing_required_pred':sum(1 for f in flags if f['missing_required_pred']>0)}
    (out/'diagnostic_summary.json').write_text(json.dumps(s,indent=2)); print(json.dumps(s,indent=2))

if __name__=='__main__': main()
