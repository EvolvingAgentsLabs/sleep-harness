# P0 cross-modelo: H-SEC-1 (firewall subconsciente) sobre Gemma 4 E4B.
# Inference-only (sin fine-tuning). ¿La firma de seguridad del pizarrón separa
# payloads maliciosos ofuscados de sus gemelos benignos en Gemma, como en Qwen
# (p=0.0195)? Decide si la limitación de Gemma era específica de números o si
# el mecanismo de veto es Qwen-específico.
import os, sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['SLEEP_MODEL'] = 'gemma4e4b'
os.environ['SLEEP_CUDA'] = '1'
LAB = os.environ.get('SLEEP_LAB', '/content/lab')
os.makedirs('/content/resultados', exist_ok=True)
sys.path.insert(0, f'{LAB}/jacobian-lens')
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'

# ejecuta el mismo exp4, ahora sobre gemma4e4b en cuda
exec(open(f'{LAB}/sleep-harness/experiments/exp4_security_firewall.py').read())
