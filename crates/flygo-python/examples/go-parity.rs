//! Direct Rust caller used to qualify the Python boundary on real actor games.
use go_actors::{Config, Pool};
use serde_json::json;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn main() {
    let config: Config = serde_json::from_str(
        &std::fs::read_to_string(
            std::env::args()
                .nth(1)
                .expect("usage: go-parity CONFIG.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let actions = config.actions();
    let games = config.games;
    let mut pool = Pool::new(config).unwrap();
    let mut requests = Vec::new();
    let mut records = Vec::new();
    for network in 0..32 {
        let mut batch = pool.start(network).unwrap();
        loop {
            requests.push(json!({"round": batch.round, "network": batch.network,
                                "features": bits(&batch.features), "active": batch.active}));
            if batch.active_count() == 0 {
                break;
            }
            batch = pool
                .evaluate(
                    batch.round,
                    batch.network,
                    &vec![0.0; games * actions],
                    &vec![0.0; games],
                )
                .unwrap();
        }
        let completed = pool.commit().unwrap();
        records.push(json!({"games": completed,
            "rows": completed.iter().flat_map(|g| g.rows.iter()).map(|r| json!({
                "features": bits(&r.features), "policy": bits(&r.policy),
                "action": r.action, "network": r.network,
                "root_value": r.root_value.to_bits(), "black_to_play": r.black_to_play
            })).collect::<Vec<_>>()}));
    }
    println!("{}", json!({"requests": requests, "commits": records}));
    pool.close();
}
