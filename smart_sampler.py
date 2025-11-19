"""
Enhanced Smart Sampler with TOP-10 Elite Focus and Large First Round

Key improvements:
- TOP 10 ELITE molecules (highest scores) get 10x exploration focus
- Large first round (4x normal size) for thorough initial exploration
- 65% elite exploitation / 35% random exploration
- Components from top-10 elites heavily prioritized
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
    """Enhanced learner with TOP-10 ELITE focus"""
    
    def __init__(self):
        # Elite storage
        self.elite_molecules = []  # All elites (up to 300)
        
        # TOP 10 ELITE - These are the absolute champions
        self.top_10_elite = []  # Top 10 highest scoring molecules
        
        # Component tracking
        self.elite_components = {
            'A': defaultdict(float),  # All elite components with weighted scores
            'B': defaultdict(float),
            'C': defaultdict(float)
        }
        
        # TOP 10 components - these get SPECIAL priority
        self.top_10_components = {
            'A': defaultdict(float),  # Components from top 10 with 10x weight
            'B': defaultdict(float),
            'C': defaultdict(float)
        }
        
        self.seen_inchikeys = set()
        
        # Strategy parameters
        self.elite_frac = 0.65           # 65% elite, 35% random
        self.mutation_prob = 0.12         # 12% mutation rate
        self.top_10_focus = 0.70         # 70% of elites come from top-10 components
        
        # Iteration tracking
        self.iteration = 0
        self.score_history = []
        
    def update_elites(self, molecules_with_scores):
        """Update elite pool with TOP-10 special focus"""
        # Add new molecules
        for name, score, inchikey in molecules_with_scores:
            if inchikey not in self.seen_inchikeys:
                self.elite_molecules.append((name, score, inchikey))
                self.seen_inchikeys.add(inchikey)
        
        # Sort by score and keep top 300
        self.elite_molecules.sort(key=lambda x: x[1], reverse=True)
        self.elite_molecules = self.elite_molecules[:300]
        
        # Extract TOP 10 elite molecules
        self.top_10_elite = self.elite_molecules[:10]
        
        # Extract components with weighted frequencies
        self._extract_elite_components()
        
        if self.elite_molecules:
            bt.logging.info(f"🏆 Elite pool updated:")
            bt.logging.info(f"   Total elites: {len(self.elite_molecules)}")
            bt.logging.info(f"   TOP 10 ELITE threshold: score ≥ {self.top_10_elite[-1][1]:.6f}")
            bt.logging.info(f"   Best score: {self.elite_molecules[0][1]:.6f}")
            bt.logging.info(f"   Top-10 components: A={len(self.top_10_components['A'])}, "
                           f"B={len(self.top_10_components['B'])}, "
                           f"C={len(self.top_10_components['C'])}")
    
    def _extract_elite_components(self):
        """Extract components with TOP-10 getting 10x weight"""
        self.elite_components = {
            'A': defaultdict(float),
            'B': defaultdict(float),
            'C': defaultdict(float)
        }
        self.top_10_components = {
            'A': defaultdict(float),
            'B': defaultdict(float),
            'C': defaultdict(float)
        }
        
        # Process TOP 10 elite molecules (10x weight)
        for name, score, _ in self.top_10_elite:
            try:
                parts = name.split(":")
                if len(parts) >= 4:
                    a_id = int(parts[2])
                    b_id = int(parts[3])
                    
                    # TOP 10 components get 10x weight
                    weight = score * 10.0
                    
                    self.top_10_components['A'][a_id] += weight
                    self.top_10_components['B'][b_id] += weight
                    
                    # Also add to general elite components
                    self.elite_components['A'][a_id] += weight
                    self.elite_components['B'][b_id] += weight
                    
                    if len(parts) > 4:
                        c_id = int(parts[4])
                        self.top_10_components['C'][c_id] += weight
                        self.elite_components['C'][c_id] += weight
                        
            except (ValueError, IndexError) as e:
                bt.logging.warning(f"Failed to parse top-10 elite: {name}")
                continue
        
        # Process remaining elites (11-300) with normal weight
        for name, score, _ in self.elite_molecules[10:]:
            try:
                parts = name.split(":")
                if len(parts) >= 4:
                    a_id = int(parts[2])
                    b_id = int(parts[3])
                    
                    # Regular elites get 1x weight
                    self.elite_components['A'][a_id] += score
                    self.elite_components['B'][b_id] += score
                    
                    if len(parts) > 4:
                        c_id = int(parts[4])
                        self.elite_components['C'][c_id] += score
                        
            except (ValueError, IndexError) as e:
                continue
        
        # Log top-10 component stats
        if self.top_10_elite:
            bt.logging.info(f"   Top-10 component extraction:")
            for role in ['A', 'B', 'C']:
                if self.top_10_components[role]:
                    top_component = max(self.top_10_components[role].items(), key=lambda x: x[1])
                    bt.logging.info(f"      Role {role}: Component {top_component[0]} (weight={top_component[1]:.2f})")
    
    def adapt_parameters(self, duplicate_ratio, score_improvement):
        """Adapt parameters based on performance"""
        self.score_history.append(score_improvement)
        
        # Calculate recent trend
        if len(self.score_history) >= 3:
            recent_trend = sum(self.score_history[-3:]) / 3
        else:
            recent_trend = score_improvement
        
        # Adaptive strategy
        if duplicate_ratio > 0.6:  # Too many duplicates
            self.mutation_prob = min(0.4, self.mutation_prob * 1.4)
            self.elite_frac = max(0.45, self.elite_frac * 0.88)
            bt.logging.info(f"⬆️ High duplicates ({duplicate_ratio:.1%}) - Increasing exploration")
            bt.logging.info(f"   mut={self.mutation_prob:.3f}, elite={self.elite_frac:.3f}")
            
        elif duplicate_ratio < 0.15:  # Too random
            self.mutation_prob = max(0.05, self.mutation_prob * 0.85)
            self.elite_frac = min(0.80, self.elite_frac * 1.1)
            bt.logging.info(f"⬇️ Low duplicates ({duplicate_ratio:.1%}) - Increasing exploitation")
            bt.logging.info(f"   mut={self.mutation_prob:.3f}, elite={self.elite_frac:.3f}")
            
        # If scores plateauing, add exploration
        elif recent_trend < 0.001 and self.iteration > 5:
            self.mutation_prob = min(0.3, self.mutation_prob * 1.2)
            bt.logging.info(f"📉 Score plateau - Adding exploration (mut={self.mutation_prob:.3f})")
        
        # If improving rapidly, maintain exploitation
        elif recent_trend > 0.01:
            self.elite_frac = min(0.75, self.elite_frac * 1.05)
            bt.logging.info(f"📈 Strong improvement - Maintaining exploitation (elite={self.elite_frac:.3f})")


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
                num_rot = rotatable_bonds_cached(smi)
                if num_rot is None:
                    continue
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
    Generate offspring with TOP-10 ELITE focus
    
    70% of elite offspring use top-10 components
    30% use general elite components
    """
    
    if not learner.elite_molecules:
        return []
    
    # Calculate how many offspring come from top-10 vs general elites
    n_top_10_offspring = int(n * learner.top_10_focus)
    n_general_offspring = n - n_top_10_offspring
    
    offspring = []
    
    # Get component pools
    pool_A_ids = [mol[0] for mol in molecules_A]
    pool_B_ids = [mol[0] for mol in molecules_B]
    pool_C_ids = [mol[0] for mol in molecules_C] if is_3_component else []
    
    # Weighted selection function
    def weighted_choice(component_dict, pool_ids):
        """Select component with probability proportional to weight"""
        if not component_dict:
            return random.choice(pool_ids)
        
        elite_ids = list(component_dict.keys())
        weights = [component_dict[comp_id] for comp_id in elite_ids]
        return random.choices(elite_ids, weights=weights, k=1)[0]
    
    local_seen = set()
    
    # Generate TOP-10 focused offspring (70%)
    bt.logging.info(f"   Generating {n_top_10_offspring} TOP-10 elite offspring")
    attempts = 0
    max_attempts = n_top_10_offspring * 10
    
    while len([o for o in offspring if o.startswith('TOP10')]) < n_top_10_offspring and attempts < max_attempts:
        attempts += 1
        
        # Decide mutation per component
        use_mutA = not learner.top_10_components['A'] or random.random() < learner.mutation_prob
        use_mutB = not learner.top_10_components['B'] or random.random() < learner.mutation_prob
        use_mutC = not learner.top_10_components['C'] or random.random() < learner.mutation_prob
        
        # Select from TOP-10 components (or mutate to random)
        if learner.top_10_components['A'] and not use_mutA:
            A = weighted_choice(learner.top_10_components['A'], pool_A_ids)
        else:
            A = random.choice(pool_A_ids)
        
        if learner.top_10_components['B'] and not use_mutB:
            B = weighted_choice(learner.top_10_components['B'], pool_B_ids)
        else:
            B = random.choice(pool_B_ids)
        
        if is_3_component:
            if learner.top_10_components['C'] and not use_mutC:
                C = weighted_choice(learner.top_10_components['C'], pool_C_ids)
            else:
                C = random.choice(pool_C_ids)
            name = f"rxn:{rxn_id}:{A}:{B}:{C}"
        else:
            name = f"rxn:{rxn_id}:{A}:{B}"
        
        # Check duplicates
        if name in local_seen:
            continue
        
        try:
            smiles = get_smiles_from_reaction_cached(name)
            if smiles:
                mol = mol_from_smiles_cached(smiles)
                if mol:
                    key = Chem.MolToInchiKey(mol)
                    if key in learner.seen_inchikeys:
                        continue
        except Exception:
            pass
        
        local_seen.add(name)
        offspring.append(f"TOP10_{name}")  # Tag as top-10 derived
    
    # Generate general elite offspring (30%)
    bt.logging.info(f"   Generating {n_general_offspring} general elite offspring")
    
    # Get top 50 from general elite components
    elite_A = sorted(learner.elite_components['A'].keys(), 
                    key=learner.elite_components['A'].get, reverse=True)[:50]
    elite_B = sorted(learner.elite_components['B'].keys(),
                    key=learner.elite_components['B'].get, reverse=True)[:50]
    elite_C = sorted(learner.elite_components['C'].keys(),
                    key=learner.elite_components['C'].get, reverse=True)[:50] if is_3_component else []
    
    attempts = 0
    max_attempts = n_general_offspring * 10
    
    while len([o for o in offspring if not o.startswith('TOP10')]) < n_general_offspring and attempts < max_attempts:
        attempts += 1
        
        # Decide mutation
        use_mutA = not elite_A or random.random() < learner.mutation_prob
        use_mutB = not elite_B or random.random() < learner.mutation_prob
        use_mutC = not elite_C or random.random() < learner.mutation_prob
        
        # Select from general elites
        if elite_A and not use_mutA:
            A = weighted_choice(learner.elite_components['A'], pool_A_ids)
        else:
            A = random.choice(pool_A_ids)
        
        if elite_B and not use_mutB:
            B = weighted_choice(learner.elite_components['B'], pool_B_ids)
        else:
            B = random.choice(pool_B_ids)
        
        if is_3_component:
            if elite_C and not use_mutC:
                C = weighted_choice(learner.elite_components['C'], pool_C_ids)
            else:
                C = random.choice(pool_C_ids)
            name = f"rxn:{rxn_id}:{A}:{B}:{C}"
        else:
            name = f"rxn:{rxn_id}:{A}:{B}"
        
        # Check duplicates
        if name in local_seen:
            continue
        
        try:
            smiles = get_smiles_from_reaction_cached(name)
            if smiles:
                mol = mol_from_smiles_cached(smiles)
                if mol:
                    key = Chem.MolToInchiKey(mol)
                    if key in learner.seen_inchikeys:
                        continue
        except Exception:
            pass
        
        local_seen.add(name)
        offspring.append(name)
    
    # Remove TOP10_ tags before returning
    cleaned_offspring = [o.replace('TOP10_', '') for o in offspring]
    
    return cleaned_offspring


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
    """Generate and validate batch"""
    
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
        bt.logging.error(f"No molecules found for roles")
        return [], [], 0.0
    
    # Generate candidates
    candidates = []
    
    # Elite offspring (65% or current elite_frac)
    if learner.elite_molecules:
        n_elite = int(n_samples * learner.elite_frac)
        elite_candidates = generate_elite_offspring(
            n_elite, rxn_id, molecules_A, molecules_B, molecules_C, 
            is_3_component, learner
        )
        candidates.extend(elite_candidates)
        bt.logging.info(f"   Generated {len(elite_candidates)} elite offspring ({learner.elite_frac:.1%})")
    
    # Random samples (35% or remaining)
    n_random = n_samples - len(candidates)
    random_candidates = generate_random_samples(
        n_random, rxn_id, molecules_A, molecules_B, molecules_C, is_3_component
    )
    candidates.extend(random_candidates)
    bt.logging.info(f"   Generated {len(random_candidates)} random samples ({1-learner.elite_frac:.1%})")
    
    # Get SMILES
    candidate_smiles = []
    for name in candidates:
        try:
            smiles = get_smiles_from_reaction_cached(name)
            candidate_smiles.append(smiles)
        except Exception:
            candidate_smiles.append(None)
    
    # Validate
    valid_names, valid_smiles = validate_smiles_sampler(candidates, candidate_smiles, subnet_config)
    
    # Deduplicate by InChIKey
    final_names = []
    final_smiles = []
    seen_in_batch = set()
    
    for name, smiles in zip(valid_names, valid_smiles):
        try:
            mol = mol_from_smiles_cached(smiles)
            if not mol:
                continue
            key = Chem.MolToInchiKey(mol)
            
            if key in seen_in_batch or key in learner.seen_inchikeys:
                continue
            
            seen_in_batch.add(key)
            learner.seen_inchikeys.add(key)
            final_names.append(name)
            final_smiles.append(smiles)
        except Exception:
            continue
    
    # Calculate duplicate ratio
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
    Enhanced smart sampler with TOP-10 elite focus and large first round
    
    First iteration: Generates 4x molecules for thorough exploration
    Later iterations: 65% elite (70% from top-10) + 35% random
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
    
    # LARGE FIRST ROUND: Generate 4x molecules on first iteration
    if _learner.iteration == 1:
        actual_n_samples = n_samples * 4
        bt.logging.info(f"🚀 FIRST ROUND - Generating 4x molecules for thorough exploration")
    else:
        actual_n_samples = n_samples
    
    bt.logging.info(f"🧬 Enhanced Smart Sampler - Iteration {_learner.iteration}")
    bt.logging.info(f"  Reaction: {rxn_id}")
    bt.logging.info(f"  Target: {actual_n_samples} molecules")
    
    if _learner.top_10_elite:
        bt.logging.info(f"  🏆 TOP 10 ELITE status active")
        bt.logging.info(f"     Threshold: score ≥ {_learner.top_10_elite[-1][1]:.6f}")
        bt.logging.info(f"     Top-10 exploitation: {_learner.top_10_focus:.1%} of elite offspring")
    else:
        bt.logging.info(f"  Elite pool: {len(_learner.elite_molecules)} molecules")
    
    bt.logging.info(f"  Elite/Random split: {_learner.elite_frac:.1%}/{1-_learner.elite_frac:.1%}")
    bt.logging.info(f"  Mutation probability: {_learner.mutation_prob:.3f}")
    
    # Generate and validate
    valid_names, valid_smiles, dup_ratio = generate_and_validate_batch(
        rxn_id, actual_n_samples, db_path, subnet_config, _learner
    )
    
    bt.logging.info(f"  ✅ Generated: {len(valid_names)}/{actual_n_samples} valid unique molecules")
    bt.logging.info(f"  📊 Duplicate ratio: {dup_ratio:.3f}")
    
    # Adapt parameters
    score_improvement = 0.0  # Will be updated by miner
    _learner.adapt_parameters(dup_ratio, score_improvement)
    
    # For first round, trim to requested size (keep best diversity)
    if _learner.iteration == 1 and len(valid_names) > n_samples:
        bt.logging.info(f"  ✂️ Trimming first round from {len(valid_names)} to {n_samples} molecules")
        # Keep every Nth molecule to maintain diversity
        step = len(valid_names) // n_samples
        valid_names = valid_names[::step][:n_samples]
        valid_smiles = valid_smiles[::step][:n_samples]
    
    # Pad with None if needed
    while len(valid_names) < n_samples:
        valid_names.append(None)
        valid_smiles.append(None)
    
    # Trim to exact count
    valid_names = valid_names[:n_samples]
    valid_smiles = valid_smiles[:n_samples]
    
    # Format output
    sampler_data = {
        "molecules": valid_names,
        "smiles": valid_smiles
    }
    
    # Save if requested
    if save_to_file and output_path:
        with open(output_path, "w") as f:
            json.dump(sampler_data, f, ensure_ascii=False, indent=2)
    
    return sampler_data


def update_learner_with_scores(molecules_with_scores: List[Tuple[str, float]]):
    """
    Update learner with scored molecules
    
    Now identifies TOP 10 elite molecules automatically
    """
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
        
        if _learner.top_10_elite:
            bt.logging.info(f"   🏆 TOP 10 ELITE molecules identified:")
            for i, (name, score, _) in enumerate(_learner.top_10_elite[:3], 1):
                bt.logging.info(f"      #{i}: {score:.6f} - {name}")


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
