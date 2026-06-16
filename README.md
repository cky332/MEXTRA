This repo contains the source code of the paper accepted by ACL'2025 main - [**"Unveiling Privacy Risks in LLM Agent Memory"**](https://arxiv.org/pdf/2502.13172).

🚩 If possible, could you please star this project. ⭐ ↗️

## 1. Thanks
The repository is partially based on [EHRAgent](https://github.com/wshi83/EhrAgent) and   [RAP](https://github.com/PanasonicConnect/rap).

## 2. Attacking on EHRAgent
You can run the following files in sequence to conduct MEXTRA on EHRAgent. 
- _EHRAgent/attacking/init_memory.py_
  
  Initialize your memory. (Optional, you can directly use our provided memory.)

- _EHRAgent/attacking/attacking_prompt_generation.py_
  
  Automatically generate attacking prompts. (Optional, you can directly use our generated memory.)
- _EHRAgent/attacking/run_attack.py_
  
  Running memory extraction attack.
- _EHRAgent/attacking/evaluation.py_
  
  Evaluation.

## 🔬 Clean reproduction + critical notes

The original code above is research-grade and not runnable as-is (hard-coded
paths, placeholder keys, needs MIMIC-III + a live WebShop server + GPT-4o).
A from-scratch, **dependency-free, one-command** reproduction of the whole
MEXTRA method — plus a code-grounded review of its strengths and weaknesses —
was added:

- [`reproduction/`](reproduction/) — pure-Python re-implementation (retrieval,
  attack-prompt design, automated generation, EN/RN/EE/CER/AER metrics). Runs
  offline in seconds: `cd reproduction && python run_demo.py`.
- [`笔记_MEXTRA.md`](笔记_MEXTRA.md) — 论文复现笔记，含从代码出发的核心优缺点分析.

## 3. Attacking on RAP
You can run the following files in sequence to conduct MEXTRA on RAP (webshop).
- _RAP/webshop/main.py_
  
  Initialize your memory. After you get the memory, set "Success" to 'true' and "Reward" to 1.0 for efficiency. (Optional, you can directly use our provided memory.)

- _RAP/attacking/attacking_prompt_generation.py_
  
  Automatically generate attacking prompts. (Optional, you can directly use our generated memory.)
- _RAP/attacking/run_attack.py_
  
  Running memory extraction attack.
- _RAP/attacking/evaluation.ipynb_

  Evaluation.


  

