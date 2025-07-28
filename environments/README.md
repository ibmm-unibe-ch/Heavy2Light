# Environment Setup
Each model is deployed in its own conda environment to isolate dependencies:
## HeavyBERTa
Uses the environment defined in [mlm_env.txt](mlm_env.txt).
## LightGPT
Runs in the environment defined in [clm_env.txt](clm_env.txt).
## Heavy2Light
Configured for adapter-based training. Create its environment with:  
```
conda create --name adapter_env --file adapter_env.txt
```
## Antibody folding ([Chai-1](https://github.com/chaidiscovery/chai-lab))
For antibody sequence folding using Chai-1, we use the environment defined in [chai_env.txt](chai_env.txt).
