# python_server/spiral_fit.py

import numpy as np

# --- アルゴリズム用の設定項目 ---
NUM_GENERATIONS = 100
NUM_CANDIDATES = 300
N_ELITES = 15
MUTATION_RATE = 0.25
MIN_A_THRESHOLD = 15.0
# 黄金螺旋の理論的なbの値
GOLDEN_B = 0.30635 
# 黄金比らしさをどれだけ重視するか
B_PENALTY_WEIGHT = 9000.0 

def calculate_composition_score(candidate_params, points, image_shape):
    h, w = image_shape
    cx, cy, a, b = candidate_params['cx'], candidate_params['cy'], candidate_params['a'], candidate_params['b']
    theta = np.linspace(-np.pi * 4, np.pi * 4, 200)
    r = a * np.exp(b * theta)
    x_fit = cx + r * np.cos(theta)
    y_fit = cy + r * np.sin(theta)
    valid_mask = (x_fit >= 0) & (x_fit < w) & (y_fit >= 0) & (y_fit < h)
    spiral_points = np.vstack((x_fit[valid_mask], y_fit[valid_mask])).T
    if len(spiral_points) < 10:
        return float('inf')
    total_distance = 0
    for p in points:
        distances = np.sqrt(np.sum((spiral_points - p)**2, axis=1))
        total_distance += np.min(distances)
    return total_distance / len(points)

def optimize_spiral_with_golden_ratio(points, image_shape):
    """
    黄金比を優先する最適化関数
    """
    h, w = image_shape
    search_ranges = {'cx': [-w, 2*w], 'cy': [-h, 2*h], 'a': [10.0, 400.0], 'b': [0.1, 0.5]}
    
    candidates = []
    for _ in range(NUM_CANDIDATES):
        candidates.append({
            'cx': np.random.uniform(*search_ranges['cx']),
            'cy': np.random.uniform(*search_ranges['cy']),
            'a': np.random.uniform(*search_ranges['a']),
            'b': np.random.uniform(*search_ranges['b'])
        })

    best_overall_params = {}
    best_overall_score = float('inf')

    for generation in range(NUM_GENERATIONS):
        scores = []
        for params in candidates:
            if params['a'] < MIN_A_THRESHOLD:
                scores.append(float('inf'))
                continue
            
            distance_score = calculate_composition_score(params, points, (h, w))
            b_penalty = (params['b'] - GOLDEN_B)**2
            final_score = distance_score + B_PENALTY_WEIGHT * b_penalty
            scores.append(final_score)
        
        sorted_candidates = sorted(zip(scores, candidates), key=lambda x: x[0])
        if sorted_candidates[0][0] < best_overall_score:
            best_overall_score = sorted_candidates[0][0]
            best_overall_params = sorted_candidates[0][1]
        
        elites = [c for s, c in sorted_candidates[:N_ELITES]]
        next_generation = elites[:]
        
        num_mutations = int(NUM_CANDIDATES * MUTATION_RATE)
        for _ in range(num_mutations):
            next_generation.append({
                'cx': np.random.uniform(*search_ranges['cx']),
                'cy': np.random.uniform(*search_ranges['cy']),
                'a': np.random.uniform(*search_ranges['a']),
                'b': np.random.uniform(*search_ranges['b'])
            })
            
        num_offspring = NUM_CANDIDATES - len(next_generation)
        for i in range(num_offspring):
            parent = elites[i % len(elites)]
            next_generation.append({
                'cx': np.random.normal(parent['cx'], w * 0.1),
                'cy': np.random.normal(parent['cy'], h * 0.1),
                'a': np.random.normal(parent['a'], 20.0),
                'b': np.random.normal(parent['b'], 0.05)
            })
        
        candidates = next_generation

        if (generation + 1) % 10 == 0:
            print(f"世代 {generation+1}/{NUM_GENERATIONS}: ベストスコア = {best_overall_score:.4f} (b={best_overall_params.get('b', 0):.4f})")
    
    return best_overall_params