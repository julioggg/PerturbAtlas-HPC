#!/usr/bin/env python3
"""Structured Correction Selector v2.

Fixes a key misspecification in Selector v1:
the target is utility RELATIVE TO the validation-selected global correction,
therefore the model must know both the candidate expert and the global baseline.

This version adds:
1. candidate descriptors;
2. global baseline descriptors;
3. candidate-minus-global relative descriptors;
4. candidate- and global-matched half-A PCA geometry;
5. a custom perturbation-expert interaction model (PEARL ranker) alongside
   tree baselines.

PEARL = Perturbation-Expert Adaptive Regret Learner.
It is an experimental architecture, not yet claimed as novel in the manuscript.

Protocol remains nested leave-one-dataset-out on the four supported core
CRISPRi essentiality datasets. Outer datasets are never used for model
selection, scaling, or epoch selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CORE=[
    "replogle_k562","replogle_rpe1","nadig25hepg2","nadig25jurkat"
]

NON_FEATURE={
    "dataset","seed","perturbation","n_cells_total",
    "oracle_regret_pearson","global_pearson","oracle_pearson",
    "target_regret_gt_0_01","target_regret_gt_0_02","appearance_weight",
}

KGRID=[8,16,32,64,128]

def base_feature_cols(df):
    return [
        c for c in df.columns
        if c not in NON_FEATURE
        and not any(t in c.lower() for t in ["oracle","regret","target"])
    ]

def matched_geom(row,k,prefix):
    if int(k)<=0:
        return {
            f"{prefix}_residual_fraction":0.0,
            f"{prefix}_cosine":1.0,
            f"{prefix}_score_l2":0.0,
        }
    kk=int(k)
    if kk not in KGRID:
        # Landscape itself should use grid values; medians such as 96 only
        # arise later during repeated-appearance aggregation.
        return {
            f"{prefix}_residual_fraction":np.nan,
            f"{prefix}_cosine":np.nan,
            f"{prefix}_score_l2":np.nan,
        }
    return {
        f"{prefix}_residual_fraction":row.get(f"pca{kk}_residual_fraction",np.nan),
        f"{prefix}_cosine":row.get(f"pca{kk}_cosine",np.nan),
        f"{prefix}_score_l2":row.get(f"pca{kk}_score_l2",np.nan),
    }

def assemble(features_path, landscape_root, regret_root):
    feat=pd.read_csv(features_path)
    feat=feat[feat.dataset.isin(CORE)].copy()
    bf=base_feature_cols(feat)
    rows=[]
    for ds in CORE:
        fds=feat[feat.dataset==ds]
        for seed in sorted(fds.seed.unique()):
            fs=fds[fds.seed==seed][["dataset","seed","perturbation"]+bf].copy()
            lp=landscape_root/ds/f"seed_{int(seed)}"/"correction_landscape.csv"
            rp=regret_root/ds/"per_perturbation_regret.csv"
            if not lp.exists() or not rp.exists():
                continue
            land=pd.read_csv(lp)
            reg=pd.read_csv(rp)
            reg=reg[reg.seed==seed][[
                "seed","perturbation","global_pearson",
                "global_pca_components","global_alpha",
                "oracle_pearson","oracle_regret_pearson"
            ]]
            x=(land.merge(fs,on=["seed","perturbation"],how="inner")
                   .merge(reg,on=["seed","perturbation"],how="inner"))
            x["delta_vs_global"]=x["pearson"]-x["global_pearson"]
            x["candidate_alpha"]=x["alpha"].astype(float)
            x["candidate_k"]=x["pca_components"].astype(float)
            x["candidate_log2_k"]=np.where(x.candidate_k>0,np.log2(x.candidate_k),0.0)
            x["global_alpha_feature"]=x["global_alpha"].astype(float)
            x["global_k_feature"]=x["global_pca_components"].astype(float)
            x["global_log2_k"]=np.where(
                x.global_k_feature>0,np.log2(x.global_k_feature),0.0
            )
            x["delta_alpha"]=x["candidate_alpha"]-x["global_alpha_feature"]
            x["abs_delta_alpha"]=x["delta_alpha"].abs()
            x["delta_log2_k"]=x["candidate_log2_k"]-x["global_log2_k"]
            x["abs_delta_log2_k"]=x["delta_log2_k"].abs()
            x["same_alpha"] = np.isclose(x.candidate_alpha,x.global_alpha_feature).astype(int)
            x["same_k"]=(x.candidate_k==x.global_k_feature).astype(int)
            x["same_config"]=(x.same_alpha.astype(bool)&x.same_k.astype(bool)).astype(int)
            x["candidate_is_identity"]=(x.candidate_alpha==0).astype(int)

            # matched geometry per row
            cg=[]; gg=[]
            for _,r in x.iterrows():
                cg.append(matched_geom(r,r["candidate_k"],"candidate"))
                gg.append(matched_geom(r,r["global_k_feature"],"global"))
            x=pd.concat([x.reset_index(drop=True),
                         pd.DataFrame(cg),pd.DataFrame(gg)],axis=1)
            for nm in ["residual_fraction","cosine","score_l2"]:
                x[f"delta_{nm}"]=x[f"candidate_{nm}"]-x[f"global_{nm}"]
            rows.append(x)

    if not rows:
        raise SystemExit("No rows assembled")
    long=pd.concat(rows,ignore_index=True)

    structured=[
        "candidate_alpha","candidate_k","candidate_log2_k",
        "global_alpha_feature","global_k_feature","global_log2_k",
        "delta_alpha","abs_delta_alpha","delta_log2_k","abs_delta_log2_k",
        "same_alpha","same_k","same_config","candidate_is_identity",
        "candidate_residual_fraction","candidate_cosine","candidate_score_l2",
        "global_residual_fraction","global_cosine","global_score_l2",
        "delta_residual_fraction","delta_cosine","delta_score_l2",
    ]
    predictors=bf+structured
    return long,bf,structured,predictors

def weights(df):
    key=df[["dataset","perturbation"]].astype(str).agg("||".join,axis=1)
    vc=key.value_counts()
    w=key.map(lambda k:1.0/vc[k]).to_numpy(float)
    dc=df.groupby("dataset")["perturbation"].nunique()
    w*=df.dataset.map(lambda d:1.0/dc[d]).to_numpy(float)
    return w*len(w)/w.sum()

def choose(df,pred):
    q=df[[
        "dataset","seed","perturbation","pca_components","alpha",
        "pearson","rmse","global_pearson","global_pca_components","global_alpha",
        "oracle_pearson","oracle_regret_pearson"
    ]].copy()
    q["predicted_delta_vs_global"]=pred
    q=q.sort_values(
        ["dataset","seed","perturbation","predicted_delta_vs_global","alpha","pca_components"],
        ascending=[True,True,True,False,True,True]
    )
    c=q.groupby(["dataset","seed","perturbation"],as_index=False).first()
    c["realized_gain_over_global"]=c.pearson-c.global_pearson
    c["remaining_oracle_regret"]=c.oracle_pearson-c.pearson
    c["beats_global"]=c.realized_gain_over_global>0
    return c

def summary(c):
    return {
        "median_gain":float(c.realized_gain_over_global.median()),
        "mean_gain":float(c.realized_gain_over_global.mean()),
        "fraction_beats":float(c.beats_global.mean()),
        "q10":float(c.realized_gain_over_global.quantile(.10)),
        "q90":float(c.realized_gain_over_global.quantile(.90)),
        "remaining_regret":float(c.remaining_oracle_regret.median()),
    }

def tree_model(kind,seed):
    if kind=="extra_v2":
        return Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",ExtraTreesRegressor(
                n_estimators=500,min_samples_leaf=8,max_features="sqrt",
                n_jobs=-1,random_state=seed
            ))
        ])
    if kind=="hgb_v2":
        return Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",HistGradientBoostingRegressor(
                learning_rate=.05,max_iter=350,max_leaf_nodes=15,
                l2_regularization=1.0,random_state=seed
            ))
        ])
    raise ValueError(kind)

class PEARL(nn.Module):
    def __init__(self,dp,de,latent=32):
        super().__init__()
        self.p=nn.Sequential(
            nn.Linear(dp,96),nn.GELU(),nn.LayerNorm(96),
            nn.Linear(96,latent)
        )
        self.e=nn.Sequential(
            nn.Linear(de,48),nn.GELU(),nn.LayerNorm(48),
            nn.Linear(48,latent)
        )
        self.bias=nn.Sequential(
            nn.Linear(dp+de,64),nn.GELU(),nn.Linear(64,1)
        )
        self.scale=math.sqrt(latent)
    def forward(self,p,e):
        zp=self.p(p); ze=self.e(e)
        return (zp*ze).sum(-1)/self.scale + self.bias(torch.cat([p,e],-1)).squeeze(-1)

class PearlWrapper:
    def __init__(self,base_cols,expert_cols,seed=20261026,epochs=40,device=None):
        self.base_cols=base_cols; self.expert_cols=expert_cols
        self.seed=seed; self.epochs=epochs
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.imp_p=SimpleImputer(strategy="median")
        self.imp_e=SimpleImputer(strategy="median")
        self.sc_p=RobustScaler()
        self.sc_e=RobustScaler()
        self.net=None

    def _fit_transform(self,df):
        p=self.sc_p.fit_transform(self.imp_p.fit_transform(df[self.base_cols]))
        e=self.sc_e.fit_transform(self.imp_e.fit_transform(df[self.expert_cols]))
        return p.astype("float32"),e.astype("float32")

    def _transform(self,df):
        p=self.sc_p.transform(self.imp_p.transform(df[self.base_cols]))
        e=self.sc_e.transform(self.imp_e.transform(df[self.expert_cols]))
        return p.astype("float32"),e.astype("float32")

    def fit(self,df,epochs=None):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        p,e=self._fit_transform(df)
        y=(df.delta_vs_global.to_numpy(float)*100.0).astype("float32")
        w=weights(df).astype("float32")
        ds=TensorDataset(
            torch.tensor(p),torch.tensor(e),torch.tensor(y),torch.tensor(w)
        )
        dl=DataLoader(ds,batch_size=4096,shuffle=True)
        self.net=PEARL(p.shape[1],e.shape[1]).to(self.device)
        opt=torch.optim.AdamW(self.net.parameters(),lr=1e-3,weight_decay=1e-4)
        lossfn=nn.SmoothL1Loss(reduction="none",beta=.5)
        ne=epochs or self.epochs
        self.net.train()
        for _ in range(ne):
            for bp,be,by,bw in dl:
                bp,be,by,bw=[x.to(self.device) for x in (bp,be,by,bw)]
                pred=self.net(bp,be)
                loss=(lossfn(pred,by)*bw).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.0)
                opt.step()
        return self

    def predict(self,df):
        p,e=self._transform(df)
        self.net.eval()
        out=[]
        with torch.no_grad():
            for i in range(0,len(df),8192):
                bp=torch.tensor(p[i:i+8192],device=self.device)
                be=torch.tensor(e[i:i+8192],device=self.device)
                out.append(self.net(bp,be).cpu().numpy())
        return np.concatenate(out)/100.0

def fit_predict(kind,tr,va,predictors,bf,structured,seed,pearl_epochs=40):
    if kind in {"extra_v2","hgb_v2"}:
        model=tree_model(kind,seed)
        model.fit(tr[predictors],tr.delta_vs_global,m__sample_weight=weights(tr))
        return model.predict(va[predictors]),model
    if kind=="pearl":
        model=PearlWrapper(bf,structured,seed=seed,epochs=pearl_epochs)
        model.fit(tr)
        return model.predict(va),model
    raise ValueError(kind)

def inner_select(train,predictors,bf,structured,seed):
    kinds=["extra_v2","hgb_v2","pearl"]
    rows=[]
    for kind in kinds:
        fold=[]
        for held in sorted(train.dataset.unique()):
            tr=train[train.dataset!=held].copy()
            va=train[train.dataset==held].copy()
            pred,_=fit_predict(kind,tr,va,predictors,bf,structured,seed)
            c=choose(va,pred); s=summary(c); s["held"]=held; fold.append(s)
        d=pd.DataFrame(fold)
        rows.append({
            "model":kind,
            "median_inner_gain":float(d.median_gain.median()),
            "mean_inner_gain":float(d.mean_gain.mean()),
            "median_fraction_beats":float(d.fraction_beats.median()),
            "worst_inner_gain":float(d.median_gain.min()),
            "median_remaining_regret":float(d.remaining_regret.median()),
        })
    tab=pd.DataFrame(rows).sort_values(
        ["median_inner_gain","mean_inner_gain","median_fraction_beats","worst_inner_gain"],
        ascending=[False,False,False,False]
    ).reset_index(drop=True)
    return str(tab.iloc[0].model),tab

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,default=Path("results/split_safe_gate_features/split_safe_gate_table.csv"))
    ap.add_argument("--landscape-root",type=Path,default=Path("results/correction_landscapes"))
    ap.add_argument("--regret-root",type=Path,default=Path("results/oracle_regret"))
    ap.add_argument("--output-dir",type=Path,default=Path("results/correction_selector_v2"))
    ap.add_argument("--seed",type=int,default=20261026)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    long,bf,structured,predictors=assemble(args.features,args.landscape_root,args.regret_root)
    outer=[]; preds=[]; sels=[]
    for held in CORE:
        tr=long[long.dataset!=held].copy()
        te=long[long.dataset==held].copy()
        best,tab=inner_select(tr,predictors,bf,structured,args.seed)
        tab.insert(0,"outer_held_out_dataset",held); sels.append(tab)
        pred,model=fit_predict(best,tr,te,predictors,bf,structured,args.seed)
        c=choose(te,pred)
        c["held_out_dataset"]=held; c["selected_model"]=best
        preds.append(c)
        s=summary(c)
        s["held_out_dataset"]=held; s["selected_model"]=best; outer.append(s)

    outer=pd.DataFrame(outer)
    pd.concat(preds,ignore_index=True).to_csv(args.output_dir/"outer_selected_configs.csv",index=False)
    pd.concat(sels,ignore_index=True).to_csv(args.output_dir/"inner_model_selection.csv",index=False)
    outer.to_csv(args.output_dir/"outer_metrics.csv",index=False)

    out={
        "method":"structured selector v2 with explicit global-baseline and matched-geometry features",
        "candidate_models":["extra_v2","hgb_v2","pearl"],
        "custom_model":"PEARL (Perturbation-Expert Adaptive Regret Learner)",
        "custom_model_status":"experimental; do not claim novelty until full literature review and external validation",
        "n_candidate_rows":int(len(long)),
        "n_predictors_tree":len(predictors),
        "n_base_halfA_features":len(bf),
        "n_structured_expert_features":len(structured),
        "median_outer_gain":float(outer.median_gain.median()),
        "mean_outer_gain":float(outer.mean_gain.mean()),
        "median_fraction_beats":float(outer.fraction_beats.median()),
        "median_remaining_regret":float(outer.remaining_regret.median()),
        "protocol":"nested leave-one-dataset-out",
        "uses_halfB_as_predictor":False,
        "uses_oracle_as_predictor":False,
    }
    (args.output_dir/"summary.json").write_text(json.dumps(out,indent=2))
    print("\n=== STRUCTURED SELECTOR V2 ===")
    print(json.dumps(out,indent=2))
    print("\n=== OUTER METRICS ===")
    print(outer.to_string(index=False))

if __name__=="__main__":
    main()
