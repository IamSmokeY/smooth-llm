import random
import string

class Perturbation:

    """Base class for random perturbations."""

    def __init__(self, q):
        self.q = q
        self.alphabet = string.printable
        print(f"Initialized Perturbation with q={q}")

class RandomSwapPerturbation(Perturbation):

    """Implementation of random swap perturbations.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2."""

    def __init__(self, q):
        super(RandomSwapPerturbation, self).__init__(q)

    def __call__(self, s):
        print(f"Applying RandomSwapPerturbation to: {s[:30]}...")

        list_s = list(s)
        num_swaps = int(len(s) * self.q / 100)
        sampled_indices = random.sample(range(len(s)), num_swaps)
        print(f"Sampled indices for swaps: {sampled_indices}")

        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)

        perturbed_string = ''.join(list_s)
        print(f"Perturbed string: {perturbed_string[:30]}...")
        return perturbed_string

class RandomPatchPerturbation(Perturbation):

    """Implementation of random patch perturbations.
    See `RandomPatchPerturbation` in lines 6-10 of Algorithm 2."""

    def __init__(self, q):
        super(RandomPatchPerturbation, self).__init__(q)

    def __call__(self, s):
        print(f"Applying RandomPatchPerturbation to: {s[:30]}...")

        list_s = list(s)
        substring_width = int(len(s) * self.q / 100)
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        print(f"Start index for patch: {start_index}")

        sampled_chars = ''.join([
            random.choice(self.alphabet) for _ in range(substring_width)
        ])
        print(f"Sampled characters for patch: {sampled_chars}")

        list_s[start_index:start_index+substring_width] = sampled_chars
        perturbed_string = ''.join(list_s)
        print(f"Perturbed string: {perturbed_string[:30]}...")
        return perturbed_string

class RandomInsertPerturbation(Perturbation):

    """Implementation of random insert perturbations.
    See `RandomInsertPerturbation` in lines 11-17 of Algorithm 2."""

    def __init__(self, q):
        super(RandomInsertPerturbation, self).__init__(q)

    def __call__(self, s):
        print(f"Applying RandomInsertPerturbation to: {s[:30]}...")

        list_s = list(s)
        num_inserts = int(len(s) * self.q / 100)
        sampled_indices = random.sample(range(len(s)), num_inserts)
        print(f"Sampled indices for inserts: {sampled_indices}")

        for i in sampled_indices:
            list_s.insert(i, random.choice(self.alphabet))

        perturbed_string = ''.join(list_s)
        print(f"Perturbed string: {perturbed_string[:30]}...")
        return perturbed_string






