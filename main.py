import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs

def main(args):

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Dynamically adjust paths
    args.results_dir = os.path.join(script_dir, args.results_dir)
    args.attack_logfile = os.path.join(script_dir, args.attack_logfile)

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)

    # Instantiate the targeted LLM
    from SafeDecoding.exp.helper import SafeDecodingManager
    safeDecodingLLM = SafeDecodingManager()

    # Create SmoothLLM instance
    defense = defenses.SmoothLLM(
        target_model=safeDecodingLLM,
        pert_type=args.smoothllm_pert_type,
        pert_pct=args.smoothllm_pert_pct,
        num_copies=args.smoothllm_num_copies
    )

    # Create attack instance, used to create prompts
    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=safeDecodingLLM
    )

    jailbroken_results = []
    outputs = []
    for i, prompt in tqdm(enumerate(attack.prompts)):
        output = defense(prompt)
        outputs.append(output)
        jb = defense.is_jailbroken(output)
        jailbroken_results.append(jb)
    import json

    # Create output file path
    output_file_path = os.path.join(args.results_dir, 'output.json')

    # Append each output to the JSON file
    with open(output_file_path, 'w') as output_file:
        output_json = {"outputs": outputs}
        output_file.write(json.dumps(output_json,indent=4))
    num_errors = len([res for res in jailbroken_results if res])
    print(f'We made {num_errors} errors')

    # Save results to a pandas DataFrame
    summary_df = pd.DataFrame.from_dict({
        'Number of smoothing copies': [args.smoothllm_num_copies],
        'Perturbation type': [args.smoothllm_pert_type],
        'Perturbation percentage': [args.smoothllm_pert_pct],
        'JB percentage': [np.mean(jailbroken_results) * 100],
        'Trial index': [args.trial]
    })
    summary_df.to_pickle(os.path.join(
        args.results_dir, 'summary.pd'
    ))
    print(summary_df)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0
    )

    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='vicuna',
        choices=['vicuna', 'llama2','llama3.1']
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='GCG',
        choices=['GCG', 'PAIR']
    )
    parser.add_argument(
        '--attack_logfile',
        type=str,
        default='data/GCG/vicuna_behaviors.json'
    )

    # SmoothLLM
    parser.add_argument(
        '--smoothllm_num_copies',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--smoothllm_pert_pct',
        type=int,
        default=10
    )
    parser.add_argument(
        '--smoothllm_pert_type',
        type=str,
        default='RandomSwapPerturbation',
        choices=[
            'RandomSwapPerturbation',
            'RandomPatchPerturbation',
            'RandomInsertPerturbation'
        ]
    )

    args = parser.parse_args()
    main(args)
