"""Prepare fixed new benchmark splits; preserve official tests and temporal separation."""
import argparse,csv,datetime,io,json,zipfile
from pathlib import Path
import numpy as np
from scripts.prepare_data import read_ts,write,sha

CLASSIFICATION={'natops':'NATOPS','epilepsy':'Epilepsy','racketsports':'RacketSports',
                'pendigits':'PenDigits','articulary':'ArticularyWordRecognition'}

def classification(raw,out,name,archive):
    with zipfile.ZipFile(raw/(archive+'.zip')) as z:
        data={};mapping=None
        for split in ('TRAIN','TEST'):
            member=next(n for n in z.namelist() if n.endswith('_'+split+'.ts'))
            x,labels=read_ts(z.read(member))
            if mapping is None:mapping={c:i for i,c in enumerate(sorted(set(labels)))}
            y=np.array([mapping[c] for c in labels]);ids=np.array([f'{split}:{i}' for i in range(len(y))]);groups=np.full(len(y),-1)
            if split=='TRAIN':
                rng=np.random.default_rng(1701);valid=[]
                for c in np.unique(y):
                    rows=np.flatnonzero(y==c);valid.extend(rng.permutation(rows)[:max(1,len(rows)//5)])
                mask=np.isin(np.arange(len(y)),valid)
                data['train']=(x[~mask],y[~mask],ids[~mask],groups[~mask]);data['val']=(x[mask],y[mask],ids[mask],groups[mask])
            else:data['test']=(x,y,ids,groups)
    write(out,name,data,dict(task='classification',source=f'UEA {archive}',source_sha256=sha(raw/(archive+'.zip')),
                            classes=mapping,protocol='official TEST; stratified 20% TRAIN validation seed1701',
                            notes='Subject identifiers unavailable; no subject-independent validation claim'))


def causal_window(x):
    """Fill only from earlier observed values within this input window, never across split/window."""
    observed=np.isfinite(x);filled=x.copy()
    for t in range(1,len(filled)):
        missing=~np.isfinite(filled[t]);filled[t,missing]=filled[t-1,missing]
    if not np.isfinite(filled).all():return None
    return np.concatenate([filled,observed.astype(float)],axis=1)


def regression(raw,out,name):
    with zipfile.ZipFile(raw/(name+'.zip')) as z:
        member=next(n for n in z.namelist() if n.endswith('.csv'));text=z.read(member).decode('utf-8-sig')
    rows=list(csv.DictReader(io.StringIO(text),delimiter=';' if name=='air_quality' else ','))
    if name=='beijing_pm25':
        columns=['pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir'];wind=['NE','NW','SE','cv']
        def value(r,c):return float(r[c]) if r[c] not in ('NA','') else np.nan
        x=np.array([[value(r,c) for c in columns]+[float(r['cbwd']==w) for w in wind] for r in rows])
        times=[datetime.datetime(*(int(r[c]) for c in ('year','month','day','hour'))) for r in rows]
        columns+=['wind_'+w for w in wind];target=x[:,0].copy();target_name='mean future six-hour PM2.5 concentration'
    else:
        rows=[r for r in rows if r.get('Date','').strip()]
        columns=['CO(GT)','PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']
        x=np.array([[float(r[c].replace(',','.')) for c in columns] for r in rows]);x[x==-200]=np.nan
        target=x[:,0].copy();target_name='mean future six-hour CO concentration (mg/m^3)'
        times=[datetime.datetime.strptime(r['Date']+' '+r['Time'],'%d/%m/%Y %H.%M.%S') for r in rows]
    seconds=np.array([int((t-datetime.datetime(1970,1,1)).total_seconds()) for t in times])
    if np.any(np.diff(seconds)<=0):raise ValueError('Non-monotonic or duplicate timestamps')
    data={s:[[],[],[],[]] for s in ('train','val','test')};dropped={'gap':0,'target_missing':0,'input_prefix_missing':0};used=[]
    for domain,chunk in enumerate(np.array_split(np.arange(len(x)),5)):
        parts=np.split(chunk,[int(.6*len(chunk)),int(.8*len(chunk))])
        for split,part in zip(data,parts):
            for offset in range(0,len(part)-29,30):
                start=int(part[offset]);end=start+30
                if not np.all(np.diff(seconds[start:end])==3600):dropped['gap']+=1;continue
                if not np.isfinite(target[start+24:end]).all():dropped['target_missing']+=1;continue
                window=causal_window(x[start:start+24])
                if window is None:dropped['input_prefix_missing']+=1;continue
                data[split][0].append(window.T);data[split][1].append([target[start+24:end].mean()])
                data[split][2].append(f'{start}:{end}');data[split][3].append(domain);used.extend(range(start,end))
    assert len(used)==len(set(used)),'Raw time overlap'
    data={s:tuple(np.array(v) for v in values) for s,values in data.items()}
    for split in data:
        assert set(data[split][3])==set(range(5)),f'Missing domain in {split}'
    write(out,name,data,dict(task='regression',source=f'UCI {381 if name=="beijing_pm25" else 360}',
                            source_sha256=sha(raw/(name+'.zip')),columns=columns+[c+'_observed' for c in columns],
                            target=target_name,raw_rows=len(x),dropped_windows=dropped,
                            protocol='5 chronological domains, raw 60/20/20 split before lookback24/horizon6/stride30',
                            imputation='causal forward-fill strictly inside each input window; leading missing input or any missing target => discard window',
                            raw_overlap=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'prepared';raw=a.root/'raw/round3'
    for name,archive in CLASSIFICATION.items():
        if (out/f'{name}_manifest.json').exists():raise FileExistsError(name)
        classification(raw,out,name,archive)
    for name in ('beijing_pm25','air_quality'):
        if (out/f'{name}_manifest.json').exists():raise FileExistsError(name)
        regression(raw,out,name)
