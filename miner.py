#!/usr/bin/env python3
"""
Competition Miner with Smart Sampling

Integrates elite recombination and adaptive mutation for progressive learning.
Reads from /workspace/input.json, writes to /output/result.json.

Key features:
- Smart sampling with elite component recombination
- Parallel PSICHIC scoring for maximum speed
- Progressive learning that improves over iterations
- Continuous output updates for timeout protection
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import bittensor as bt
import pandas as pd
from rdkit import Chem
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup paths - COMPETITION STANDARD
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Competition imports - USE NOVA_PH2 PACKAGE
import nova_ph2
from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction
from smart_sampler import run_sampler, update_learner_with_scores

# Database path - USE NOVA_PH2 PACKAGE
DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

# Output paths - COMPETITION REQUIREMENT
OUTPUT_PATH = "/output/result.json"
SAMPLER_PATH = "/tmp/sampler_file.json"


def get_config(input_file="/workspace/input.json"):
    """Get config from competition input file"""
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config


def calculate_final_scores(
    score_dict: dict, 
    sampler_data: dict, 
    config: dict
) -> pd.DataFrame:
    """
    Calculate final weighted scores per molecule.
    
    Formula: avg_target - (antitarget_weight × avg_antitarget)
    
    Args:
        score_dict: Scores with 'ps_target_scores' and 'ps_antitarget_scores'
        sampler_data: Generated molecules {"molecules": [...], "smiles": [...]}
        config: Config with antitarget_weight
    
    Returns:
        DataFrame with columns: name, smiles, InChIKey, score
    """
    names = sampler_data["molecules"]
    smiles = sampler_data["smiles"]
    
    # Calculate InChIKey for deduplication
    inchikey_list = []
    for s in smiles:
        try:
            if s:
                inchikey_list.append(Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
            else:
                inchikey_list.append(None)
        except Exception as e:
            bt.logging.error(f"Error calculating InChIKey: {e}")
            inchikey_list.append(None)
    
    # Extract scores
    targets = score_dict['ps_target_scores']
    antitargets = score_dict['ps_antitarget_scores']
    
    # Calculate final weighted scores
    final_scores = []
    for mol_idx in range(len(names)):
        try:
            # Average target scores across all target proteins
            target_scores_for_mol = [target_list[mol_idx] for target_list in targets]
            avg_target = sum(target_scores_for_mol) / len(target_scores_for_mol)
            
            # Average antitarget scores across all antitarget proteins
            antitarget_scores_for_mol = [antitarget_list[mol_idx] for antitarget_list in antitargets]
            avg_antitarget = sum(antitarget_scores_for_mol) / len(antitarget_scores_for_mol)
            
            # Final weighted score
            score = avg_target - (config["antitarget_weight"] * avg_antitarget)
            final_scores.append(score)
        except Exception as e:
            bt.logging.error(f"Error calculating score for molecule {mol_idx}: {e}")
            final_scores.append(float('-inf'))
    
    # Create dataframe
    batch_scores = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikey_list,
        "score": final_scores
    })
    
    return batch_scores


def iterative_sampling_loop(
    db_path: str,
    sampler_file_path: str,
    output_path: str,
    config: dict
) -> None:
    """
    Main mining loop with smart sampling and progressive learning.
    
    Process:
    1. Initialize PSICHIC models once at start
    2. Smart sample N molecules (elite recombination + random exploration)
    3. Score with PSICHIC models in parallel (FAST!)
    4. Calculate weighted scores
    5. Update elite learner (teaches sampler for next iteration)
    6. Merge with top pool and deduplicate
    7. Write best molecules to output immediately
    
    Loop continues until competition timeout (typically 1800s).
    """
    
    # Initialize PSICHIC models once at start
    bt.logging.info(f"[Miner] 🔬 Initializing PSICHIC models...")
    target_models = []
    antitarget_models = []
    
    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)
        bt.logging.info(f"[Miner]   ✓ Loaded target model for {seq}")
    
    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)
        bt.logging.info(f"[Miner]   ✓ Loaded antitarget model for {seq}")
    
    bt.logging.info(f"[Miner] ✅ Models ready: {len(target_models)} targets, {len(antitarget_models)} antitargets\n")
    
    n_samples = config["num_molecules"] * 5  # Generate 5x for better selection
    
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    iteration = 0
    while True:
        iteration += 1
        bt.logging.info(f"\n{'='*70}")
        bt.logging.info(f"[Miner] 🔄 ITERATION {iteration}")
        bt.logging.info(f"{'='*70}")
        
        # STEP 1: Smart sampling with elite recombination
        bt.logging.info(f"[Miner] Generating {n_samples} molecules with smart sampler...")
        sampler_data = run_sampler(
            n_samples=n_samples,
            subnet_config=config,
            output_path=sampler_file_path,
            save_to_file=True,
            db_path=db_path,
        )
        
        if not sampler_data or not sampler_data.get("molecules"):
            bt.logging.warning("[Miner] ⚠️ No valid molecules produced; continuing")
            continue
        
        valid_count = len([m for m in sampler_data["molecules"] if m])
        bt.logging.info(f"[Miner] ✅ Generated {valid_count} valid molecules")
        
        # Filter to only valid molecules with smiles
        filtered_names = []
        filtered_smiles = []
        for name, smile in zip(sampler_data["molecules"], sampler_data["smiles"]):
            if name and smile:
                filtered_names.append(name)
                filtered_smiles.append(smile)
        
        if not filtered_names:
            bt.logging.warning("[Miner] ⚠️ No valid molecules with SMILES; continuing")
            continue
        
        sampler_data = {"molecules": filtered_names, "smiles": filtered_smiles}
        
        # STEP 2: Score with PSICHIC models in PARALLEL (FAST!)
        bt.logging.info(f"[Miner] 🔬 Scoring {len(filtered_smiles)} molecules with {len(target_models)} target + {len(antitarget_models)} antitarget models (parallel)...")
        
        def score_with_model(model, smiles_list):
            """Helper function for parallel scoring"""
            try:
                res = model.score_molecules(smiles_list)
                return res['predicted_binding_affinity'].tolist()
            except Exception as e:
                bt.logging.error(f"[Miner] Model scoring failed: {e}")
                # Return zeros to avoid breaking the iteration
                return [0.0] * len(smiles_list)
        
        # Parallel scoring of target models
        all_target_results = []
        with ThreadPoolExecutor(max_workers=max(1, len(target_models))) as executor:
            futures = {executor.submit(score_with_model, m, filtered_smiles): idx 
                      for idx, m in enumerate(target_models)}
            for fut in as_completed(futures):
                all_target_results.append(fut.result())
        
        # Parallel scoring of antitarget models
        all_antitarget_results = []
        with ThreadPoolExecutor(max_workers=max(1, len(antitarget_models))) as executor:
            futures = {executor.submit(score_with_model, m, filtered_smiles): idx 
                      for idx, m in enumerate(antitarget_models)}
            for fut in as_completed(futures):
                all_antitarget_results.append(fut.result())
        
        score_dict = {
            'ps_target_scores': all_target_results,
            'ps_antitarget_scores': all_antitarget_results,
        }
        
        bt.logging.info(f"[Miner] ✅ Parallel scoring complete")
        
        # STEP 3: Calculate final weighted scores
        batch_scores = calculate_final_scores(score_dict, sampler_data, config)
        
        # Filter out invalid molecules
        batch_scores = batch_scores[batch_scores['name'].notna()]
        batch_scores = batch_scores[batch_scores['score'] > float('-inf')]
        
        if batch_scores.empty:
            bt.logging.warning("[Miner] ⚠️ No valid scores; continuing")
            continue
        
        bt.logging.info(f"[Miner] 📊 Scored {len(batch_scores)} molecules:")
        bt.logging.info(f"         Best score: {batch_scores['score'].max():.6f}")
        bt.logging.info(f"         Mean score: {batch_scores['score'].mean():.6f}")
        
        # STEP 4: Teach the learner with scored molecules
        bt.logging.info(f"[Miner] 🧠 Teaching learner with scores...")
        molecules_with_scores = [
            (row['name'], row['score']) 
            for _, row in batch_scores.iterrows()
        ]
        update_learner_with_scores(molecules_with_scores)
        
        # STEP 5: Merge with existing pool
        top_pool = pd.concat([top_pool, batch_scores], ignore_index=True)
        
        # Sort by score BEFORE deduplication (keeps highest scoring duplicate)
        top_pool = top_pool.sort_values(by="score", ascending=False)
        
        # Deduplicate by InChIKey (keep first occurrence = highest score)
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        
        # Keep top N
        top_pool = top_pool.head(config["num_molecules"])
        
        # STEP 6: Write to output immediately (for timeout protection)
        top_entries = {"molecules": top_pool["name"].tolist()}
        
        with open(output_path, "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)
        
        bt.logging.info(f"[Miner] ✅ ITERATION {iteration} COMPLETE:")
        bt.logging.info(f"         Wrote {len(top_entries['molecules'])} molecules to {output_path}")
        bt.logging.info(f"         Top pool best: {top_pool['score'].max():.6f}")
        bt.logging.info(f"         Top pool mean: {top_pool['score'].mean():.6f}")
        bt.logging.info(f"         Total unique molecules: {len(top_pool)}")


def main(config: dict):
    """Main entry point"""
    bt.logging.info("\n" + "="*70)
    bt.logging.info("🚀 COMPETITION MINER WITH SMART SAMPLING + PARALLEL SCORING")
    bt.logging.info("="*70)
    bt.logging.info(f"📋 Configuration:")
    bt.logging.info(f"   Target proteins: {config.get('target_sequences', [])}")
    bt.logging.info(f"   Antitarget proteins: {config.get('antitarget_sequences', [])}")
    bt.logging.info(f"   Allowed reaction: {config.get('allowed_reaction', 'unknown')}")
    bt.logging.info(f"   Target molecules: {config.get('num_molecules', 100)}")
    bt.logging.info(f"   Antitarget weight: {config.get('antitarget_weight', 0.9)}")
    bt.logging.info(f"   Min heavy atoms: {config.get('min_heavy_atoms', 10)}")
    bt.logging.info(f"   Rotatable bonds: {config.get('min_rotatable_bonds', 0)}-{config.get('max_rotatable_bonds', 50)}")
    bt.logging.info("="*70 + "\n")
    
    iterative_sampling_loop(
        db_path=DB_PATH,
        sampler_file_path=SAMPLER_PATH,
        output_path=OUTPUT_PATH,
        config=config
    )


if __name__ == "__main__":
    config = get_config()
    main(config)
