import numpy as np
import pandas as pd

SIGNALS=['meas_v','meas_t1','meas_t2','thermal_t','meas_p','meas_ch4',
         'meas_co','meas_co2','meas_hcn','meas_hcl','meas_hf','meas_n2o',
         'meas_no','meas_no2','meas_so2','atmp_t','indu_t','indu_t.1']

def event_features(frame):
    n=len(frame)
    raw=frame[SIGNALS].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    pieces=[raw.add_prefix('raw_'),raw.isna().astype(float).add_prefix('missing_')]
    # Features are assigned to original positions after chronological per-experiment computation.
    names={}
    def assign(name,idx,values):
        if name not in names:
            names[name]=np.full(n,np.nan)
        names[name][idx]=values
    times=frame.relative_time_s.to_numpy(float)
    for _,idx in frame.groupby('experiment_id',sort=False).indices.items():
        idx=idx[np.argsort(times[idx],kind='stable')]
        ts=times[idx]
        assert np.all(np.diff(ts)>0), 'Duplicate times need explicit simultaneous-observation semantics'
        values=raw.iloc[idx].copy()
        values.index=pd.to_timedelta(ts,unit='s')
        dt=np.r_[0.,np.diff(ts)]
        assign('observed_dt',idx,dt)
        for window in (15,60,180,600):
            rolling=values.rolling(f'{window}s',closed='right',min_periods=1)
            for stat,result in [('mean',rolling.mean()),('std',rolling.std(ddof=0)),
                                ('min',rolling.min()),('max',rolling.max())]:
                for c in SIGNALS:
                    assign(f'{c}_{stat}_{window}s',idx,result[c].to_numpy())
            lag=np.searchsorted(ts,ts-window,side='right')-1
            valid=lag>=0
            for c in SIGNALS:
                x=values[c].to_numpy(float)
                delta=np.full(len(idx),np.nan)
                rate=delta.copy()
                delta[valid]=x[valid]-x[lag[valid]]
                rate[valid]=delta[valid]/(ts[valid]-ts[lag[valid]])
                assign(f'{c}_delta_{window}s',idx,delta)
                assign(f'{c}_slope_{window}s',idx,rate)
        for c in SIGNALS:
            x=values[c].to_numpy(float)
            assign(f'{c}_initial_delta',idx,x-x[0])
            previous=pd.Series(x).shift(1)
            assign(f'{c}_step_rate',idx,np.divide(x-previous.to_numpy(),dt,out=np.full(len(idx),np.nan),where=dt>0))
            maximum=values[c].cummax().to_numpy()
            minimum=values[c].cummin().to_numpy()
            assign(f'{c}_below_observed_peak',idx,maximum-x)
            assign(f'{c}_above_observed_min',idx,x-minimum)
            if c in ('meas_t1','meas_t2','thermal_t','meas_p','meas_v','indu_t','indu_t.1'):
                peak=-np.inf
                peak_time=np.nan
                since=np.full(len(idx),np.nan)
                for j in range(len(idx)):
                    if np.isfinite(x[j]) and x[j]>=peak:
                        peak,peak_time=x[j],ts[j]
                    since[j]=ts[j]-peak_time
                assign(f'{c}_seconds_since_peak',idx,since)
        for c in ('meas_t1','meas_t2','thermal_t'):
            x=values[c].to_numpy(float)
            for threshold in (45,60,80,100,150,200,300):
                hot=x>=threshold
                first=np.nan
                age=np.full(len(idx),np.nan)
                for j in range(len(idx)):
                    if hot[j] and not np.isfinite(first):
                        first=ts[j]
                    age[j]=ts[j]-first
                assign(f'{c}_since_first_above_{threshold}',idx,age)
                assign(f'{c}_hot_duration_{threshold}',idx,np.cumsum(hot*dt))
                assign(f'{c}_degree_seconds_{threshold}',idx,np.cumsum(np.nan_to_num(np.maximum(x-threshold,0))*dt))
        volts=values.meas_v.to_numpy()
        peak=values.meas_v.cummax().to_numpy()
        for fraction in (.9,.5,.1):
            low=volts<np.maximum(peak,1e-6)*fraction
            first=np.nan
            age=np.full(len(idx),np.nan)
            for j in range(len(idx)):
                if low[j] and not np.isfinite(first):
                    first=ts[j]
                age[j]=ts[j]-first
            assign(f'voltage_since_below_peak_{fraction}',idx,age)
    new=pd.concat([*pieces,pd.DataFrame(names)],axis=1)
    new['temperature_max']=raw[['meas_t1','meas_t2','thermal_t']].max(axis=1)
    new['temperature_min']=raw[['meas_t1','meas_t2','thermal_t']].min(axis=1)
    new['t1_minus_thermal']=raw.meas_t1-raw.thermal_t
    new['t2_minus_thermal']=raw.meas_t2-raw.thermal_t
    new['heater_max']=raw[['indu_t','indu_t.1']].max(axis=1)
    new['heater_minus_t1']=new.heater_max-raw.meas_t1
    gases=[c for c in SIGNALS if c.startswith('meas_') and c not in ['meas_v','meas_t1','meas_t2','meas_p']]
    new['gas_zero_count']=raw[gases].eq(0).sum(axis=1)
    return new.replace([np.inf,-np.inf],np.nan).astype(np.float32)
