import os
os.environ['SLEEP_MODEL']='gemma4e4b'; os.environ['SLEEP_CUDA']='1'
os.environ['SLEEP_CAPAS_LO']='0.35'; os.environ['SLEEP_CAPAS_HI']='0.65'
LAB=os.environ.get('SLEEP_LAB','/content/lab')
import sys; sys.path.insert(0,f'{LAB}/jacobian-lens'); sys.path.insert(0,f'{LAB}/jlens-harness'); sys.path.insert(0,f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS']=f'{LAB}/jlens-harness'
_e=f'{LAB}/sleep-harness/experiments/exp4_security_firewall.py'
exec(compile(open(_e).read(),_e,'exec'),{'__file__':_e,'__name__':'__main__'})
