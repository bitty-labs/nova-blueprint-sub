"""
Smart Sampler - OPTIMIZED VERSION
- Removed duplicate InChIKey calculations
- Removed pre-filtering (moved to miner)
- Simplified elite strategy
"""

import sqlite3
import random
import os
import json
from typing import List, Tuple, Optional, Set
from collections import defaultdict
from functools import lru_cache
import bittensor as bt
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

import sys
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from nova_ph2.combinatorial_db.reactions import get_reaction_info, get_smiles_from_reaction
from nova_ph2.utils import get_heavy_atom_count


class EliteLearner:
    """SIMPLIFIED learner - removed top-10 complexity"""
    
    def __init__(self):
        # Single elite pool
        self.elite_molecules = []  # Up to 300
        self.seen_inchikeys = set()  # Track globally to avoid re-generation
        
        # Simple parameters
        self.elite_frac = 0.30           # 30% elite
        self.mutation_prob = 0.10        # 10% mutation
        
        # Iteration tracking
        self.iteration = 0
        self.score_history = []
        
    def update_elites(self, molecules_with_scores):
        """Update elite pool - SIMPLIFIED"""
        # Add new molecules
        for name, score, inchikey in molecules_with_scores:
            if inchikey not in self.seen_inchikeys:
                self.elite_molecules.append((name, score, inchikey))
                self.seen_inchikeys.add(inchikey)
        
        # Sort by score and keep top 300
        self.elite_molecules.sort(key=lambda x: x[1], reverse=True)
        self.elite_molecules = self.elite_molecules[:300]
        
        if self.elite_molecules:
            bt.logging.info(f"🏆 Elite pool updated:")
            bt.logging.info(f"   Total elites: {len(self.elite_molecules)}")
            bt.logging.info(f"   Best score: {self.elite_molecules[0][1]:.6f}")
    
    def adapt_parameters(self, duplicate_ratio, score_improvement):
        """Adapt parameters based on performance"""
        self.score_history.append(score_improvement)
        
        if len(self.score_history) >= 3:
            recent_trend = sum(self.score_history[-3:]) / 3
        else:
            recent_trend = score_improvement
        
        # Simple adaptation
        if duplicate_ratio > 0.6:
            self.mutation_prob = min(0.4, self.mutation_prob * 1.4)
            self.elite_frac = max(0.2, self.elite_frac * 0.9)
            bt.logging.info(f"⬆️ High duplicates ({duplicate_ratio:.1%}) - Increasing exploration")
            
        elif duplicate_ratio < 0.15:
            self.mutation_prob = max(0.05, self.mutation_prob * 0.85)
            self.elite_frac = min(0.50, self.elite_frac * 1.1)
            bt.logging.info(f"⬇️ Low duplicates ({duplicate_ratio:.1%}) - Increasing exploitation")


# Global learner instance
_learner = EliteLearner()


@lru_cache(maxsize=200_000)
def get_smiles_from_reaction_cached(name: str) -> Optional[str]:
    """Cached version of get_smiles_from_reaction"""
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None


@lru_cache(maxsize=200_000)
def mol_from_smiles_cached(s: str):
    """Cached RDKit mol creation"""
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


@lru_cache(maxsize=200_000)
def rotatable_bonds_cached(s: str) -> Optional[int]:
    """Cached rotatable bonds calculation"""
    mol = mol_from_smiles_cached(s)
    if mol is None:
        return None
    try:
        return Descriptors.NumRotatableBonds(mol)
    except Exception:
        return None


@lru_cache(maxsize=1024)
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
    """
    SIMPLIFIED elite offspring generation
    - Single loop (no top-10 vs general split)
    - No string tagging
    - Defer expensive duplicate checks
    """
    
    if not learner.elite_molecules:
        return []
    
    # Extract elite components (top 50 only)
    elite_As = set()
    elite_Bs = set()
    elite_Cs = set()
    
    for name, score, _ in learner.elite_molecules[:50]:
        try:
            parts = name.split(":")
            if len(parts) >= 4:
                elite_As.add(int(parts[2]))
                elite_Bs.add(int(parts[3]))
                if len(parts) > 4:
                    elite_Cs.add(int(parts[4]))
        except (ValueError, IndexError):
            continue
    
    # Get component pools
    pool_A_ids = [mol[0] for mol in molecules_A]
    pool_B_ids = [mol[0] for mol in molecules_B]
    pool_C_ids = [mol[0] for mol in molecules_C] if is_3_component else []
    
    # SINGLE LOOP - generate offspring
    offspring = []
    local_seen = set()
    attempts = 0
    max_attempts = n * 5
    
    while len(offspring) < n and attempts < max_attempts:
        attempts += 1
        
        # Simple mutation decision
        use_elite_A = elite_As and random.random() > learner.mutation_prob
        use_elite_B = elite_Bs and random.random() > learner.mutation_prob
        use_elite_C = elite_Cs and random.random() > learner.mutation_prob
        
        # Pick components
        A = random.choice(list(elite_As)) if use_elite_A else random.choice(pool_A_ids)
        B = random.choice(list(elite_Bs)) if use_elite_B else random.choice(pool_B_ids)
        
        if is_3_component:
            C = random.choice(list(elite_Cs)) if use_elite_C else random.choice(pool_C_ids)
            name = f"rxn:{rxn_id}:{A}:{B}:{C}"
        else:
            name = f"rxn:{rxn_id}:{A}:{B}"
        
        # CHEAP duplicate check - just string comparison
        if name in local_seen:
            continue
        
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


def generate_batch(
    rxn_id: int,
    n_samples: int,
    db_path: str,
    learner: EliteLearner
) -> Tuple[List[str], List[str]]:
    """
    Generate raw candidates with SMILES
    NO DEDUPLICATION - that happens in miner
    """
    
    # Get reaction info
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return [], []
    
    smarts, roleA, roleB, roleC = reaction_info
    is_3_component = roleC is not None and roleC != 0
    
    # Get molecule pools
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_3_component else []
    
    if not molecules_A or not molecules_B or (is_3_component and not molecules_C):
        bt.logging.error(f"No molecules found for roles")
        return [], []
    
    # Generate candidates
    candidates = []
    
    # Elite offspring
    if learner.elite_molecules:
        n_elite = int(n_samples * learner.elite_frac)
        elite_candidates = generate_elite_offspring(
            n_elite, rxn_id, molecules_A, molecules_B, molecules_C, 
            is_3_component, learner
        )
        candidates.extend(elite_candidates)
        bt.logging.info(f"   Generated {len(elite_candidates)} elite offspring ({learner.elite_frac:.1%})")
    
    # Random samples
    n_random = n_samples - len(candidates)
    random_candidates = generate_random_samples(
        n_random, rxn_id, molecules_A, molecules_B, molecules_C, is_3_component
    )
    candidates.extend(random_candidates)
    bt.logging.info(f"   Generated {len(random_candidates)} random samples ({1-learner.elite_frac:.1%})")
    
    # Get SMILES for all candidates (dedup happens in miner)
    candidate_smiles = []
    valid_candidates = []
    
    for name in candidates:
        try:
            smiles = get_smiles_from_reaction_cached(name)
            if smiles:
                valid_candidates.append(name)
                candidate_smiles.append(smiles)
        except Exception:
            continue
    
    bt.logging.info(f"   ✅ Generated {len(valid_candidates)} candidates with SMILES")
    
    return valid_candidates, candidate_smiles


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
    OPTIMIZED sampler - returns raw candidates
    All filtering happens in miner
    """
    global _learner
    
    if seed is not None:
        random.seed(seed)
    
    if subnet_config is None:
        bt.logging.error("subnet_config is required")
        return {"molecules": [], "smiles": []}
    
    # Get reaction ID
    allowed_reaction = subnet_config.get("allowed_reaction", "")
    if not allowed_reaction or not allowed_reaction.startswith("rxn:"):
        bt.logging.error(f"Invalid allowed_reaction: {allowed_reaction}")
        return {"molecules": [], "smiles": []}
    
    try:
        rxn_id = int(allowed_reaction.split(":")[-1])
    except (ValueError, IndexError):
        bt.logging.error(f"Could not parse reaction ID from: {allowed_reaction}")
        return {"molecules": [], "smiles": []}
    
    # Seed learner with external elites if provided
    if elite_names:
        bt.logging.info(f"Seeding learner with {len(elite_names)} external elites")
        for name in elite_names:
            try:
                smiles = get_smiles_from_reaction_cached(name)
                if smiles:
                    mol = mol_from_smiles_cached(smiles)
                    if mol:
                        key = Chem.MolToInchiKey(mol)
                        _learner.elite_molecules.append((name, 1.0, key))
                        _learner.seen_inchikeys.add(key)
            except Exception:
                continue
    
    # Override parameters if provided
    if elite_frac is not None:
        _learner.elite_frac = elite_frac
    if mutation_prob is not None:
        _learner.mutation_prob = mutation_prob
    if avoid_inchikeys:
        _learner.seen_inchikeys.update(avoid_inchikeys)
    
    _learner.iteration += 1
    
    # REACTION-SPECIFIC first round sizing
    if _learner.iteration == 1:
        if rxn_id == 5:  # Simple 2-component
            actual_n_samples = n_samples
            bt.logging.info(f"🚀 FIRST ROUND (rxn:5) - Standard size: {actual_n_samples}")
        else:
            actual_n_samples = n_samples * 4
            bt.logging.info(f"🚀 FIRST ROUND (rxn:{rxn_id}) - 4x size: {actual_n_samples}")
    else:
        actual_n_samples = n_samples
    
    bt.logging.info(f"🧬 Smart Sampler - Iteration {_learner.iteration}")
    bt.logging.info(f"  Reaction: {rxn_id}")
    bt.logging.info(f"  Target: {actual_n_samples} molecules")
    bt.logging.info(f"  Elite pool: {len(_learner.elite_molecules)} molecules")
    bt.logging.info(f"  Elite/Random split: {_learner.elite_frac:.1%}/{1-_learner.elite_frac:.1%}")
    bt.logging.info(f"  Mutation probability: {_learner.mutation_prob:.3f}")
    
    # Generate candidates
    candidates, candidate_smiles = generate_batch(rxn_id, actual_n_samples, db_path, _learner)
    
    # Return raw data - miner will handle all filtering
    sampler_data = {
        "molecules": candidates,
        "smiles": candidate_smiles
    }
    
    # Save if requested
    if save_to_file and output_path:
        with open(output_path, "w") as f:
            json.dump(sampler_data, f, ensure_ascii=False, indent=2)
    
    return sampler_data


def update_learner_with_scores(molecules_with_scores: List[Tuple[str, float]]):
    """Update learner with scored molecules"""
    global _learner
    
    # Calculate score improvement
    if molecules_with_scores:
        current_best = max(score for _, score in molecules_with_scores)
        previous_best = _learner.elite_molecules[0][1] if _learner.elite_molecules else 0.0
        score_improvement = current_best - previous_best
    else:
        score_improvement = 0.0
    
    # Convert to format with InChIKeys
    elite_data = []
    for name, score in molecules_with_scores:
        try:
            smiles = get_smiles_from_reaction_cached(name)
            if smiles:
                mol = mol_from_smiles_cached(smiles)
                if mol:
                    key = Chem.MolToInchiKey(mol)
                    elite_data.append((name, score, key))
        except Exception:
            continue
    
    if elite_data:
        _learner.update_elites(elite_data)
        bt.logging.info(f"🧠 Learner updated with {len(elite_data)} scored molecules")
        bt.logging.info(f"   Score improvement: {score_improvement:+.6f}")


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
