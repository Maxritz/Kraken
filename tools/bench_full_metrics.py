#!/usr/bin/env python3
"""Full-metric benchmark: interleaved arms, medians with spread, every column."""
import subprocess, re, os, statistics, json, sys
M="/home/user/fix/dense-768x12-q8.gguf"
base="The quick brown fox jumps over the lazy dog and keeps running through the field. "
PROMPTS={"short(~46)":(base.strip(),4096),
         "mid(~900)":((base*18).strip(),4096),
         "long(~2040)":((base*40).strip(),4096)}
ARMS=[("scalar",{"KRK_CPU_ATTN_SCALAR":"1"}),
      ("in-place",{"KRK_CPU_ATTN_GATHER":"0"}),
      ("gather",{})]
COMPONENTS=("attention","gemm_q4k_i8_rN","gemm_q6k_rN","gemm_q4k_i8_r1","gemm_q6k_r1",
            "silu_mul","rope","kv_append","rmsnorm","embed","gemm_f16_rN")

def one(env,P,ctx,n=8):
    e=dict(os.environ); e.update(env)
    cmd=["./build/kraken","-m",M,"--cpu","-p",P,"-n",str(n),"--greedy","--ctx",str(ctx),"--kprof"]
    p=subprocess.run(cmd,capture_output=True,env=e,timeout=3600)
    o=re.sub(rb"\x1b\[[0-9;]*m",b"",p.stderr).decode("utf-8","replace")
    r={"components":{}}
    m=re.search(r"prefill\s+(\d+) tok in ([\d.]+) ms = ([\d.]+)",o)
    if m: r["prefill"]={"tok":int(m.group(1)),"ms":float(m.group(2)),"tps":float(m.group(3))}
    m=re.search(r"decode\s+(\d+) tok in ([\d.]+) ms over (\d+) timed steps = ([\d.]+) tok/s \(([\d.]+) ms/step\)",o)
    if m: r["decode"]={"tok":int(m.group(1)),"ms":float(m.group(2)),"steps":int(m.group(3)),
                       "tps":float(m.group(4)),"ms_step":float(m.group(5))}
    for line in o.splitlines():
        f=line.split()
        if len(f)>=7 and f[0] in COMPONENTS:
            try:
                r["components"][f[0]]={"ops":float(f[1]),"dev_us":float(f[2]),"us_op":float(f[3]),
                    "pct_run":float(f[4].rstrip('%')),"gmacs":float(f[5]),
                    "pct_peak":(float(f[6].rstrip('%')) if f[6] not in ('-','') else None)}
            except ValueError: pass
    m=re.search(r"roofline probe: int8 (\d+) GMAC/s, f32 (\d+) GMAC/s",o)
    if m: r["roofline"]={"int8":float(m.group(1)),"f32":float(m.group(2))}
    return r

REPS=int(sys.argv[1]) if len(sys.argv)>1 else 5
data={}
for label,(P,ctx) in PROMPTS.items():
    for rep in range(REPS):
        for name,env in ARMS:              # interleaved: every rep touches every arm
            data.setdefault((label,name),[]).append(one(env,P,ctx))
    print(f"  {label} done", flush=True)

def med(xs): xs=[x for x in xs if x is not None]; return statistics.median(xs) if xs else None
def rng(xs):
    xs=[x for x in xs if x is not None]
    return (min(xs),max(xs)) if xs else (None,None)

out={}
for label in PROMPTS:
    print(f"\n=== {label}  ({REPS} interleaved reps per arm)")
    print(f"{'arm':10} {'prefill tok/s':>22} {'decode tok/s':>20} {'attn ms/op':>18} {'attn %run':>12} {'attn GMAC/s':>12} {'%peak':>8}")
    for name,_ in ARMS:
        rows=data[(label,name)]
        pre=[r.get("prefill",{}).get("tps") for r in rows]
        # A run with zero timed decode steps reports 0.0 tok/s, which is
        # arithmetic (prompt filled the context), not a speed. Keep it out of
        # the median and print it as missing rather than as a measurement.
        dec=[r["decode"]["tps"] for r in rows
             if r.get("decode",{}).get("steps",0)>0]
        att=[r["components"].get("attention") for r in rows]
        ams=[a["us_op"]/1000.0 for a in att if a]
        apc=[a["pct_run"] for a in att if a]
        agm=[a["gmacs"] for a in att if a]
        apk=[a["pct_peak"] for a in att if a and a["pct_peak"] is not None]
        pm,pM=rng(pre); dm,dM=rng(dec); am,AM=rng(ams)
        dtxt = (f"{med(dec):7.1f} [{dm:5.1f},{dM:5.1f}]" if dec else
                f"{'n/a':>7} {'(no timed steps)':>13}")
        print(f"{name:10} {med(pre):9.1f} [{pm:7.1f},{pM:7.1f}] {dtxt} "
              f"{med(ams):7.1f} [{am:6.1f},{AM:6.1f}] {med(apc):11.1f}% {med(agm):12.1f} {med(apk):7.1f}%")
        out[f"{label}|{name}"]={"prefill_tps":{"med":med(pre),"min":pm,"max":pM},
            "decode_tps":{"med":med(dec),"min":dm,"max":dM,
                          "measured":bool(dec)},
            "attn_ms_op":{"med":med(ams),"min":am,"max":AM},
            "attn_pct_run":med(apc),"attn_gmacs":med(agm),"attn_pct_peak":med(apk),
            "components":rows[0]["components"],"roofline":rows[0].get("roofline")}
    s=med([r.get("prefill",{}).get("tps") for r in data[(label,"scalar")]])
    ip=med([r.get("prefill",{}).get("tps") for r in data[(label,"in-place")]])
    g=med([r.get("prefill",{}).get("tps") for r in data[(label,"gather")]])
    print(f"{'ratios':10} in-place {ip/s:.2f}x scalar   gather {g/s:.2f}x scalar   gather {g/ip:.2f}x in-place")
json.dump(out,open("/tmp/full_metrics.json","w"),indent=1)
print("\nWROTE /tmp/full_metrics.json")
