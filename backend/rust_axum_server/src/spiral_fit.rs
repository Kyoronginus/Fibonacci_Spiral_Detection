use rand::Rng;
use rand::seq::SliceRandom;
use std::f64::consts::PI;
use std::cmp::Ordering;
use rayon::prelude::*;

const NUM_GENERATIONS: usize = 80;
const NUM_CANDIDATES: usize = 300;
const N_ELITES: usize = 15;
const MUTATION_RATE: f64 = 0.25;
const MIN_A_THRESHOLD: f64 = 15.0;
const GOLDEN_B: f64 = 0.30635;

#[derive(Clone, Debug)]
pub struct SpiralParams {
    pub cx: f64,
    pub cy: f64,
    pub a: f64,
    pub b: f64,
}

pub fn calculate_composition_score(params: &SpiralParams, points: &[[f64; 2]], image_shape: (u32, u32)) -> f64 {
    let (w, h) = (image_shape.0 as f64, image_shape.1 as f64);
    let theta_steps = 200;
    let max_theta = PI * 4.0;
    let min_theta = -PI * 4.0;
    
    let mut spiral_points = Vec::with_capacity(theta_steps);
    
    for i in 0..theta_steps {
        let t = min_theta + (max_theta - min_theta) * (i as f64 / (theta_steps - 1) as f64);
        let r = params.a * (params.b * t).exp();
        let x = params.cx + r * t.cos();
        let y = params.cy + r * t.sin();
        
        if x >= 0.0 && x < w && y >= 0.0 && y < h {
            spiral_points.push([x, y]);
        }
    }

    if spiral_points.len() < 10 {
        return f64::INFINITY;
    }

    let mut total_distance = 0.0;
    for point in points {
        let mut min_dist = f64::INFINITY;
        for sp in &spiral_points {
            let dist_sq = (sp[0] - point[0]).powi(2) + (sp[1] - point[1]).powi(2);
            if dist_sq < min_dist {
                min_dist = dist_sq;
            }
        }
        total_distance += min_dist.sqrt();
    }

    total_distance / points.len() as f64
}

pub fn optimize_spiral_with_golden_ratio(points: &[[f64; 2]], image_shape: (u32, u32), b_penalty_weight: f64) -> SpiralParams {
    let (w, h) = (image_shape.0 as f64, image_shape.1 as f64);
    let mut rng = rand::thread_rng();

    let mut candidates: Vec<SpiralParams> = (0..NUM_CANDIDATES).map(|_| {
        SpiralParams {
            cx: rng.gen_range(-w..2.0*w),
            cy: rng.gen_range(-h..2.0*h),
            a: rng.gen_range(10.0..400.0),
            b: rng.gen_range(0.1..0.5),
        }
    }).collect();

    let mut best_overall_params = candidates[0].clone();
    let mut best_overall_score = f64::INFINITY;

    for _generation in 0..NUM_GENERATIONS {
        let mut scored_candidates: Vec<(f64, SpiralParams)> = candidates.par_iter().map(|params| {
            if params.a < MIN_A_THRESHOLD {
                return (f64::INFINITY, params.clone());
            }

            let distance_score = calculate_composition_score(params, points, image_shape);
            let b_penalty = (params.b - GOLDEN_B).powi(2);
            let final_score = distance_score + b_penalty_weight * b_penalty;
            
            (final_score, params.clone())
        }).collect();

        scored_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        if scored_candidates[0].0 < best_overall_score {
            best_overall_score = scored_candidates[0].0;
            best_overall_params = scored_candidates[0].1.clone();
        }

        let elites: Vec<SpiralParams> = scored_candidates.iter().take(N_ELITES).map(|x| x.1.clone()).collect();
        let mut next_generation = elites.clone();

        let num_mutations = (NUM_CANDIDATES as f64 * MUTATION_RATE) as usize;
        for _ in 0..num_mutations {
            next_generation.push(SpiralParams {
                cx: rng.gen_range(-w..2.0*w),
                cy: rng.gen_range(-h..2.0*h),
                a: rng.gen_range(10.0..400.0),
                b: rng.gen_range(0.1..0.5),
            });
        }

        let num_offspring = NUM_CANDIDATES - next_generation.len();
        for i in 0..num_offspring {
            let parent = &elites[i % elites.len()];
            next_generation.push(SpiralParams {
                cx: rng.gen::<f64>() * w * 0.1 + parent.cx,
                cy: rng.gen::<f64>() * h * 0.1 + parent.cy,
                a: rng.gen::<f64>() * 20.0 + parent.a,
                b: rng.gen::<f64>() * 0.05 + parent.b,
            });
        }
        candidates = next_generation;
    }

    best_overall_params
}
