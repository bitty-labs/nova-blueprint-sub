"""
Smart Sampler with Elite Recombination and Adaptive Mutation

Drop-in replacement for random_sampler.py with intelligent learning:
- Component-level elite recombination (extracts A, B, C reactant IDs)
- Adaptive mutation rates based on duplicate detection
- Global InChIKey deduplication across all iterations
- Progressive learning from high-scoring molecules

Compatible with baseline miner.py - same interface, smarter strategy.
"""

import sqlite3
import random
import os
import json
from typing import List, Tuple, Optional, Set
from collections import defaultdict
import bittensor as bt
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

import sys
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from combinatorial_db.reactions import get_reaction_info, get_smiles_from_reaction
from utils import get_smiles, find_chemically_identical, get_heavy_atom_count


class EliteLearner:
    """Maintains elite molecules and adapts exploration strategy"""
    
    def __init__(self):
        self.elite_molecules = []  # List of (name, score, inchikey)
        self.elite_components = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
        self.seen_inchikeys = set()
        self.mutation_prob = 0.15  # Start with moderate mutation
        self.elite_frac = 0.5      # 50% elite offspring
        self.iteration = 0
        
    def update_elites(self, molecules_with_scores):
        """Update elite pool with new high-scoring molecules"""
        for name, score, inchikey in molecules_with_scores:
            if inchikey not in self.seen_inchikeys:
                self.elite_molecules.append((name, score, inchikey))
                self.seen_inchikeys.add(inchikey)
        
        # Sort by score and keep top 300
        self.elite_molecules.sort(key=lambda x: x[1], reverse=True)
        self.elite_molecules = self.elite_molecules[:300]
        
        # Extract components from elites
        self._extract_elite_components()
        
        if self.elite_molecules:
            bt.logging.info(f"Elite pool: {len(self.elite_molecules)} molecules, "
                           f"top score: {self.elite_molecules[0][1]:.6f}")
    
    def _extract_elite_components(self):
        """Extract reactant IDs from elite molecules"""
        self.elite_components = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
        
        for name, score, _ in self.elite_molecules:
            try:
                parts = name.split(":")
                if len(parts) >= 4:
                    self.elite_components['A'][int(parts[2])] += 1
                    self.elite_components['B'][int(parts[3])] += 1
                    if len(parts) > 4:
                        self.elite_components['C'][int(parts[4])] += 1
            except (ValueError, IndexError):
                continue
    
    def adapt_parameters(self, duplicate_ratio):
        """Adapt exploration based on duplicate rate"""
        if duplicate_ratio > 0.6:  # Too repetitive
            self.mutation_prob = min(0.5, self.mutation_prob * 1.3)
            self.elite_frac = max(0.2, self.elite_frac * 0.85)
            bt.logging.info(f"⬆️ Increasing exploration: mut={self.mutation_prob:.3f}, elite={self.elite_frac:.3f}")
        elif duplicate_ratio < 0.2:  # Too random
            self.mutation_prob = max(0.05, self.mutation_prob * 0.85)
            self.elite_frac = min(0.8, self.elite_frac * 1.15)
            bt.logging.info(f"⬇️ Increasing exploitation: mut={self.mutation_prob:.3f}, elite={self.elite_frac:.3f}")


# Global learner instance - persists across iterations
_learner = EliteLearner()


def validate_smiles_sampler(names: List[str], smiles_list: List[Optional[str]], config: dict) -> Tuple[List[str], List[str]]:
    """Validate molecules based on heavy atoms and rotatable bonds"""
    valid_names: List[str] = []
    valid_smiles: List[str] = []
    
    for name, smi in zip(names, smiles_list):
        try:
            if not smi:
                continue
            
            # Check heavy atoms
            if get_heavy_atom_count(smi) < config.get('min_heavy_atoms', 10):
                continue
            
            # Check rotatable bonds
            try:
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    continue
                num_rot = Descriptors.NumRotatableBonds(mol)
                if num_rot < config.get('min_rotatable_bonds', 0) or \
                   num_rot > config.get('max_rotatable_bonds', 50):
                    continue
            except Exception:
                continue
            
            valid_names.append(name)
            valid_smiles.append(smi)
        except Exception:
            continue
    
    return valid_names, valid_smiles


def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    """Get all molecules that match the specified role_mask"""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
            (role_mask, role_mask)
        )
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []


def generate_elite_offspring(
    n: int,
    rxn_id: int,
    molecules_A: List[Tuple],
    molecules_B: List[Tuple],
    molecules_C: List[Tuple],
    is_3_component: bool,
    learner: EliteLearner
) -> List[str]:
    """Generate offspring by recombining components from elite molecules"""
    
    if not learner.elite_molecules:
        return []
    
    # Get elite components (top 30 most frequent)
    elite_A = sorted(learner.elite_components['A'].keys(), 
                    key=learner.elite_components['A'].get, reverse=True)[:30]
    elite_B = sorted(learner.elite_components['B'].keys(),
                    key=learner.elite_components['B'].get, reverse=True)[:30]
    elite_C = sorted(learner.elite_components['C'].keys(),
                    key=learner.elite_components['C'].get, reverse=True)[:30] if is_3_component else []
    
    # Get all available IDs
    pool_A_ids = [mol[0] for mol in molecules_A]
    pool_B_ids = [mol[0] for mol in molecules_B]
    pool_C_ids = [mol[0] for mol in molecules_C] if is_3_component else []
    
    offspring = []
    attempts = 0
    max_attempts = n * 10
    local_seen = set()
    
    while len(offspring) < n and attempts < max_attempts:
        attempts += 1
        
        # Decide whether to mutate each component
        use_mutA = not elite_A or random.random() < learner.mutation_prob
        use_mutB = not elite_B or random.random() < learner.mutation_prob
        use_mutC = not elite_C or random.random() < learner.mutation_prob
        
        # Select components
        A = random.choice(pool_A_ids) if use_mutA else random.choice(elite_A)
        B = random.choice(pool_B_ids) if use_mutB else random.choice(elite_B)
        
        if is_3_component:
            C = random.choice(pool_C_ids) if use_mutC else random.choice(elite_C)
            name = f"rxn:{rxn_id}:{A}:{B}:{C}"
        else:
            name = f"rxn:{rxn_id}:{A}:{B}"
        
        # Avoid local duplicates
        if name in local_seen:
            continue
        
        # Check InChIKey to avoid global duplicates
        try:
            smiles = get_smiles_from_reaction(name)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    key = Chem.MolToInchiKey(mol)
                    if key in learner.seen_inchikeys:
                        continue
        except Exception:
            pass
        
        local_seen.add(name)
        offspring.append(name)
    
    return offspring


def generate_random_samples(
    n: int,
    rxn_id: int,
    molecules_A: List[Tuple],
    molecules_B: List[Tuple],
    molecules_C: List[Tuple],
    is_3_component: bool
) -> List[str]:
    """Generate random molecular combinations"""
    
    samples = []
    local_seen = set()
    attempts = 0
    max_attempts = n * 5
    
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        
        mol_A = random.choice(molecules_A)
        mol_B = random.choice(molecules_B)
        
        if is_3_component:
            mol_C = random.choice(molecules_C)
            name = f"rxn:{rxn_id}:{mol_A[0]}:{mol_B[0]}:{mol_C[0]}"
        else:
            name = f"rxn:{rxn_id}:{mol_A[0]}:{mol_B[0]}"
        
        if name not in local_seen:
            local_seen.add(name)
            samples.append(name)
    
    return samples


def generate_and_validate_batch(
    rxn_id: int,
    n_samples: int,
    db_path: str,
    subnet_config: dict,
    learner: EliteLearner
) -> Tuple[List[str], List[str], float]:
    """Generate and validate a batch of molecules, return duplicate ratio"""
    
    # Get reaction info
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return [], [], 0.0
    
    smarts, roleA, roleB, roleC = reaction_info
    is_3_component = roleC is not None and roleC != 0
    
    # Get molecule pools
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_3_component else []
    
    if not molecules_A or not molecules_B or (is_3_component and not molecules_C):
        bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return [], [], 0.0
    
    # Generate candidates: mix of elite offspring and random samples
    candidates = []
    
    # Elite offspring
    if learner.elite_molecules:
        n_elite = int(n_samples * learner.elite_frac)
        elite_candidates = generate_elite_offspring(
            n_elite, rxn_id, molecules_A, molecules_B, molecules_C, 
            is_3_component, learner
        )
        candidates.extend(elite_candidates)
        bt.logging.info(f"Generated {len(elite_candidates)} elite offspring")
    
    # Random samples
    n_random = n_samples - len(candidates)
    random_candidates = generate_random_samples(
        n_random, rxn_id, molecules_A, molecules_B, molecules_C, is_3_component
    )
    candidates.extend(random_candidates)
    bt.logging.info(f"Generated {len(random_candidates)} random samples")
    
    # Get SMILES for all candidates
    candidate_smiles = []
    for name in candidates:
        try:
            smiles = get_smiles_from_reaction(name)
            candidate_smiles.append(smiles)
        except Exception:
            candidate_smiles.append(None)
    
    # Validate (heavy atoms + rotatable bonds)
    valid_names, valid_smiles = validate_smiles_sampler(candidates, candidate_smiles, subnet_config)
    
    # Deduplicate by InChIKey
    final_names = []
    final_smiles = []
    seen_in_batch = set()
    
    for name, smiles in zip(valid_names, valid_smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            key = Chem.MolToInchiKey(mol)
            
            # Skip if seen in this batch or globally
            if key in seen_in_batch or key in learner.seen_inchikeys:
                continue
            
            seen_in_batch.add(key)
            learner.seen_inchikeys.add(key)
            final_names.append(name)
            final_smiles.append(smiles)
        except Exception:
            continue
    
    # Calculate duplicate ratio for adaptation
    duplicate_ratio = 1.0 - (len(final_names) / len(candidates)) if candidates else 0.0
    
    return final_names, final_smiles, duplicate_ratio


def run_sampler(
    n_samples: int = 1000,
    seed: int = None,
    subnet_config: dict = None,
    output_path: str = None,
    save_to_file: bool = False,
    db_path: str = None,
    elite_names: List[str] = None,
    elite_frac: float = None,
    mutation_prob: float = None,
    avoid_inchikeys: Set[str] = None
) -> dict:
    """
    Smart sampler with elite recombination and adaptive mutation.
    
    Drop-in replacement for random_sampler.run_sampler() with same interface.
    
    Args:
        n_samples: Number of molecules to generate
        seed: Random seed (optional)
        subnet_config: Configuration dict with validation params
        output_path: Path to save output JSON (optional)
        save_to_file: Whether to save to file
        db_path: Path to molecules database
        elite_names: External elite molecules to seed learning (optional)
        elite_frac: Override elite fraction (optional)
        mutation_prob: Override mutation probability (optional)
        avoid_inchikeys: InChIKeys to avoid (optional)
    
    Returns:
        Dictionary with format: {"molecules": ["rxn:1:2:3", ...]}
    """
    global _learner
    
    if seed is not None:
        random.seed(seed)
    
    if subnet_config is None:
        bt.logging.error("subnet_config is required")
        return {"molecules": []}
    
    # Get reaction ID from config
    allowed_reaction = subnet_config.get("allowed_reaction", "")
    if not allowed_reaction or not allowed_reaction.startswith("rxn:"):
        bt.logging.error(f"Invalid allowed_reaction: {allowed_reaction}")
        return {"molecules": []}
    
    try:
        rxn_id = int(allowed_reaction.split(":")[-1])
    except (ValueError, IndexError):
        bt.logging.error(f"Could not parse reaction ID from: {allowed_reaction}")
        return {"molecules": []}
    
    # Update learner with external elites if provided
    if elite_names:
        bt.logging.info(f"Seeding learner with {len(elite_names)} external elites")
        for name in elite_names:
            try:
                smiles = get_smiles_from_reaction(name)
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        key = Chem.MolToInchiKey(mol)
                        _learner.elite_molecules.append((name, 1.0, key))
            except Exception:
                continue
        _learner._extract_elite_components()
    
    # Override parameters if provided
    if elite_frac is not None:
        _learner.elite_frac = elite_frac
    if mutation_prob is not None:
        _learner.mutation_prob = mutation_prob
    if avoid_inchikeys:
        _learner.seen_inchikeys.update(avoid_inchikeys)
    
    _learner.iteration += 1
    bt.logging.info(f"🧬 Smart Sampler - Iteration {_learner.iteration}")
    bt.logging.info(f"  Reaction: {rxn_id}")
    bt.logging.info(f"  Target: {n_samples} molecules")
    bt.logging.info(f"  Elite pool: {len(_learner.elite_molecules)} molecules")
    bt.logging.info(f"  Mutation prob: {_learner.mutation_prob:.3f}")
    bt.logging.info(f"  Elite fraction: {_learner.elite_frac:.3f}")
    
    # Generate and validate batch
    valid_names, valid_smiles, dup_ratio = generate_and_validate_batch(
        rxn_id, n_samples, db_path, subnet_config, _learner
    )
    
    bt.logging.info(f"  ✅ Generated: {len(valid_names)}/{n_samples} valid unique molecules")
    bt.logging.info(f"  📊 Duplicate ratio: {dup_ratio:.3f}")
    
    # Adapt parameters based on duplicate ratio
    _learner.adapt_parameters(dup_ratio)
    
    # Pad with None if we didn't get enough
    while len(valid_names) < n_samples:
        valid_names.append(None)
    
    # Trim to exact count
    valid_names = valid_names[:n_samples]
    
    # Format output
    sampler_data = {"molecules": valid_names}
    
    # Save to file if requested
    if save_to_file and output_path:
        with open(output_path, "w") as f:
            json.dump(sampler_data, f, ensure_ascii=False, indent=2)
    
    return sampler_data


def update_learner_with_scores(molecules_with_scores: List[Tuple[str, float]]):
    """
    Update the learner with scored molecules from miner.
    
    Call this from miner after scoring to teach the learner.
    
    Args:
        molecules_with_scores: List of (molecule_name, score) tuples
    """
    global _learner
    
    # Convert to format with InChIKeys
    elite_data = []
    for name, score in molecules_with_scores:
        try:
            smiles = get_smiles_from_reaction(name)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    key = Chem.MolToInchiKey(mol)
                    elite_data.append((name, score, key))
        except Exception:
            continue
    
    if elite_data:
        _learner.update_elites(elite_data)
        bt.logging.info(f"🧠 Learner updated with {len(elite_data)} scored molecules")


if __name__ == "__main__":
    # Test the sampler
    test_config = {
        "allowed_reaction": "rxn:1",
        "min_heavy_atoms": 10,
        "min_rotatable_bonds": 0,
        "max_rotatable_bonds": 50
    }
    
    result = run_sampler(
        n_samples=100,
        subnet_config=test_config,
        db_path="../combinatorial_db/molecules.sqlite"
    )
    
    valid_count = len([m for m in result['molecules'] if m])
    print(f"Generated {valid_count} valid molecules")
