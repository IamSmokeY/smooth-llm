import json
import pandas as pd

class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.max_new_tokens = max_new_tokens
        print(f"Initialized Prompt with full_prompt length={len(full_prompt)} and perturbable_prompt={perturbable_prompt[:30]}...")

    def perturb(self, perturbation_fn):
        print(f"Perturbing prompt: {self.perturbable_prompt[:30]}...")
        perturbed_prompt = perturbation_fn(self.perturbable_prompt)
        self.full_prompt = self.full_prompt.replace(
            self.perturbable_prompt,
            perturbed_prompt
        )
        self.perturbable_prompt = perturbed_prompt
        print(f"Perturbed prompt: {self.perturbable_prompt[:30]}...")

class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model
        print(f"Initialized Attack with logfile={logfile}")

class GCG(Attack):

    """Greedy Coordinate Gradient attack.

    Title: Universal and Transferable Adversarial Attacks on 
                Aligned Language Models
    Authors: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
    Paper: https://arxiv.org/abs/2307.15043
    """

    def __init__(self, logfile, target_model):
        super(GCG, self).__init__(logfile, target_model)
        print(f"Initializing GCG with logfile={logfile}")

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        self.goals = log['goal']
        self.targets = log['target']
        self.controls = log['controls']

        print(f"Loaded GCG data: {len(self.goals)} goals, {len(self.controls)} controls, {len(self.targets)} targets")

        self.prompts = [
            self.create_prompt(g, c, t)
            for (g, c, t) in zip(self.goals, self.controls, self.targets)
        ]
        print(f"Created {len(self.prompts)} prompts")

    def create_prompt(self, goal, control, target, max_new_len=100):
        """Create GCG prompt."""
        print(f"Creating prompt with goal={goal[:30]}..., control={control[:30]}..., target={target[:30]}...")

        max_new_tokens = max(
            len(self.target_model.tokenizer(target).input_ids) + 2,
            max_new_len
        )

        # Create full prompt for LLM
        conv_template = self.target_model.conv_template
        conv_template.append_message(
            conv_template.roles[0], f"{goal} {control}"
        )
        conv_template.append_message(conv_template.roles[1], f"")
        prompt = conv_template.get_prompt()

        # As per the GCG source code, we encode then decode the full prompt
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode(
            encoding.input_ids
        ).replace('<s>','').replace('</s>','')
        print(f"Generated full_prompt: {full_prompt[:100]}...")

        # Clear the conv template
        conv_template.messages = []

        start_index = full_prompt.find(goal)
        end_index = full_prompt.find(control) + len(control)
        perturbable_prompt = full_prompt[start_index:end_index]
        print(f"Extracted perturbable_prompt: {perturbable_prompt[:30]}...")

        return Prompt(
            full_prompt, 
            perturbable_prompt, 
            max_new_tokens
        )

class PAIR(Attack):

    """Prompt Automatic Iterative Refinement (PAIR) attack.

    Title: Jailbreaking Black Box Large Language Models in Twenty Queries
    Authors: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, 
                George J. Pappas, Eric Wong
    Paper: https://arxiv.org/abs/2310.08419
    """

    def __init__(self, logfile, target_model):
        super(PAIR, self).__init__(logfile, target_model)
        print(f"Initializing PAIR with logfile={logfile}")

        df = pd.read_pickle(logfile)
        jailbreak_prompts = df['jailbreak_prompt'].to_list()
        
        print(f"Loaded {len(jailbreak_prompts)} jailbreak prompts")

        self.prompts = [
            self.create_prompt(prompt)
            for prompt in jailbreak_prompts
        ]
        print(f"Created {len(self.prompts)} prompts from PAIR")

    def create_prompt(self, prompt):
        print(f"Creating prompt: {prompt[:30]}...")

        conv_template = self.target_model.conv_template
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        full_prompt = conv_template.get_prompt()

        # Clear the conv template
        conv_template.messages = []

        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=100
        )

