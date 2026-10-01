"""Summarize the independent correlation-sensitivity runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/energyplus_diagnosis'


def main():
    cases=[('baseline',ROOT/'analysis/energyplus_free_running/base')]+[
        (name,OUT/name) for name in ['exterior_convection','sky','exterior_and_sky','all_convection_and_sky']]
    rows=[]; by_zone=[]
    for name,path in cases:
        record=dict(variant=name)
        for excluded in [0,7,14]:
            frame=pd.read_csv(path/f'room_metrics_exclude_{excluded}d.csv')
            record[f'room_rmse_exclude_{excluded}d']=float(np.sqrt(np.mean(frame.rmse**2)))
            record[f'room_bias_exclude_{excluded}d']=float(frame.bias.mean())
            by_zone.append(frame.assign(variant=name,excluded_days=excluded))
        surface=pd.read_csv(path/'surface_metrics_exclude_7d.csv')
        for category in ['wall','roof','floor','partition']:
            for side in ['inside','outside']:
                for q in ['Ts','cond','conv','rad']:
                    f=surface[(surface.category==category)&(surface.side==side)&(surface.quantity==q)]
                    if len(f):
                        record[f'{category}_{side}_{q}_rmse']=float(f.rmse.iloc[0])
        rows.append(record)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'causal_summary.csv',index=False)
    pd.concat(by_zone).to_csv(OUT/'room_sensitivity.csv',index=False)
    labels=['Current GraphBEM','EP exterior convection','EP sky split','EP exterior convection\n+ sky split',
            'EP interior + exterior\nconvection + sky split']
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,col,title in zip(axes,['room_rmse_exclude_7d','room_bias_exclude_7d'],['Room-temperature RMSE (°C)','Room-temperature bias (°C)']):
        bars=ax.barh(labels,frame[col],color=['.5','#4c78a8','#f58518','#54a24b','#b279a2'])
        ax.axvline(0,color='k',lw=.7)
        ax.set_title(title)
        ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.margins(x=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Free-running sensitivity, August 8–31; identical geometry, materials and weather')
    fig.tight_layout()
    fig.savefig(OUT/'causal_comparison.png',dpi=160)
    plt.close(fig)
    print(frame[['variant','room_rmse_exclude_7d','room_bias_exclude_7d','roof_outside_Ts_rmse','roof_outside_conv_rmse','roof_outside_rad_rmse']].to_string(index=False))


if __name__=='__main__':
    main()
