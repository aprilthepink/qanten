//! glTF loading: geometry, skins, morph targets, every interpolation mode,
//! and hostile files. Assets come from `tests/data/gltf/make_assets.py`.
#![cfg(feature = "gltf")]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use glam::{Mat4, Quat, Vec3};
use qanten::Npu;
use qanten::rng::SplitMix64;
use qanten_graphics::gltf::GltfScene;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/gltf")
}

fn tentacle() -> GltfScene {
    GltfScene::load(data().join("tentacle.gltf")).unwrap()
}

fn close(a: Vec3, b: Vec3, tol: f32) -> bool {
    (a - b).abs().max_element() < tol
}

#[test]
fn embedded_binary_and_external_buffers_load_the_same() {
    let reference = tentacle().mesh(0).unwrap();
    assert_eq!(reference.positions.len(), 16 * 12);
    assert_eq!(reference.triangles.len(), 15 * 12 * 2);
    assert_eq!(reference.normals.as_ref().map(Vec::len), Some(16 * 12));
    for file in ["tentacle.glb", "tentacle_external.gltf"] {
        let scene = GltfScene::load(data().join(file)).unwrap();
        assert_eq!(scene.mesh(0).unwrap(), reference, "{file}");
    }
    // Without a base folder, an external buffer cannot be resolved.
    let bytes = std::fs::read(data().join("tentacle_external.gltf")).unwrap();
    assert!(GltfScene::from_slice(&bytes, None).is_err());
}

#[test]
fn hierarchy_and_names() {
    let s = tentacle();
    assert_eq!(s.mesh_names(), [Some("Tentacle")]);
    assert_eq!(
        s.node_names(),
        [Some("Root"), Some("Mid"), Some("Tip"), Some("Body")]
    );
    assert_eq!(
        (s.parent(0), s.parent(1), s.parent(2)),
        (None, Some(0), Some(1))
    );
    assert_eq!(
        s.animation_names(),
        [Some("Bend"), Some("Hop"), Some("Pulse"), Some("Breathe")]
    );
    let bend = s.animation("Bend").unwrap();
    assert_eq!((bend.start(), bend.end(), bend.duration()), (0.0, 2.0, 2.0));
}

#[test]
fn rest_pose_skins_to_the_rest_mesh() {
    let s = tentacle();
    let model = s.skinned_mesh(0).unwrap();
    assert_eq!(model.skin.joints, [0, 1, 2]);
    assert_eq!(model.skin.name.as_deref(), Some("Chain"));
    let bones = s.rest_pose().skinning_matrices(&model.skin);
    for b in &bones {
        assert!(b.abs_diff_eq(Mat4::IDENTITY, 1e-6), "{b}");
    }
    let rest = s.mesh(0).unwrap().positions;
    let posed = model.mesh.skin(&bones).unwrap();
    assert!(
        posed
            .positions
            .iter()
            .zip(&rest)
            .all(|(a, b)| close(*a, *b, 1e-6))
    );
}

#[test]
fn linear_rotation_matches_hand_computed_skinning() {
    let s = tentacle();
    let model = s.skinned_mesh(0).unwrap();
    let bend = s.animation("Bend").unwrap();

    // LINEAR rotation: halfway between keys is the halfway angle.
    let half = bend.sample(0.5);
    let mid = half.local(1).unwrap().rotation;
    assert!(mid.abs_diff_eq(Quat::from_rotation_z(0.3), 1e-5), "{mid}");

    // The top ring is bound to the tip joint only, so it follows
    // T(0,0.7)·Rz(0.6)·T(0,0.7)·Rz(0.8) · T(0,−1.4) exactly.
    let tip = Mat4::from_translation(Vec3::Y * 0.7)
        * Mat4::from_rotation_z(0.6)
        * Mat4::from_translation(Vec3::Y * 0.7)
        * Mat4::from_rotation_z(0.8)
        * Mat4::from_translation(-Vec3::Y * 1.4);
    let bones = bend.skinning_matrices(&model.skin, 1.0);
    assert!(bones[2].abs_diff_eq(tip, 1e-5));
    let rest = s.mesh(0).unwrap().positions;
    let posed = model.mesh.skin(&bones).unwrap().positions;
    for v in rest.len() - 12..rest.len() {
        let expected = tip.transform_point3(rest[v]);
        assert!(close(posed[v], expected, 1e-5), "vertex {v}");
    }

    // The NPU path agrees with the CPU path (bf16 error on the offsets).
    let npu_model = s
        .skinned_mesh(0)
        .unwrap()
        .mesh
        .with_npu(&Npu::simulator())
        .unwrap();
    let on_npu = npu_model.skin(&bones).unwrap().positions;
    let worst = on_npu
        .iter()
        .zip(&posed)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0, f32::max);
    assert!(worst < 0.02, "NPU skinning error {worst}");
}

#[test]
fn step_and_cubic_spline_interpolation() {
    let s = tentacle();
    let hop = s.animation("Hop").unwrap();
    assert_eq!(hop.sample(0.99).local(0).unwrap().translation, Vec3::ZERO);
    assert_eq!(
        hop.sample(1.0).local(0).unwrap().translation,
        Vec3::new(0.0, 0.5, 0.0)
    );
    // Clamped outside the keyframes.
    assert_eq!(
        hop.sample(5.0).local(0).unwrap().translation,
        Vec3::new(0.0, 0.5, 0.0)
    );

    // Hermite spline with zero tangents from 1 to 2: 1 + (3s² − 2s³).
    let pulse = s.animation("Pulse").unwrap();
    for (t, want) in [(0.0, 1.0), (0.25, 1.15625), (0.5, 1.5), (1.0, 2.0)] {
        let got = pulse.sample(t).local(0).unwrap().scale;
        assert!(close(got, Vec3::splat(want), 1e-6), "t={t}: {got}");
    }
}

#[test]
fn morph_targets_become_blendshapes() {
    let s = tentacle();
    let morph = s.blendshapes(0).unwrap();
    assert_eq!(morph.default_weights, [0.25]);
    assert_eq!(morph.node, Some(3));
    let base = morph.shapes.evaluate(&[0.0]).unwrap();
    let full = morph.shapes.evaluate(&[1.0]).unwrap();
    // The bulge is 0.1·sin(πy/2) outward: 0.1 at the middle ring (y = 1)...
    let ring = |r: usize| r * 12..(r + 1) * 12;
    for v in ring(7).chain(ring(8)) {
        let d = full[v] - base[v];
        let y = base[v].y;
        let want = 0.1 * (std::f32::consts::PI * y / 2.0).sin();
        assert!((d.length() - want).abs() < 1e-5 && d.y.abs() < 1e-6);
    }
    // ...and zero at the ends.
    for v in ring(0) {
        assert!(close(full[v], base[v], 1e-6));
    }
    // The Breathe animation drives the weight.
    let breathe = s.animation("Breathe").unwrap();
    assert_eq!(breathe.sample(0.5).morph_weights(3), Some(&[0.5][..]));
    assert_eq!(s.rest_pose().morph_weights(3), None);
}

#[test]
fn missing_parts_are_errors() {
    let s = tentacle();
    assert!(s.mesh(1).is_err());
    assert!(s.skinned_mesh(1).is_err());
    assert!(s.animation("Fly").is_none());
}

// ---- hostile files -------------------------------------------------------

/// Loads `json` and exercises every reader; panics fail the test.
fn exercise(bytes: &[u8]) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Ok(s) = GltfScene::from_slice(bytes, Some(&data())) else {
            return;
        };
        let _ = s.mesh(0);
        let _ = s.blendshapes(0);
        if let Ok(m) = s.skinned_mesh(0) {
            for i in 0..4 {
                if let Some(c) = s.animation_at(i) {
                    for t in [-1.0, 0.0, 0.3, 1.0, 1.7, 9.0] {
                        let bones = c.skinning_matrices(&m.skin, t);
                        let _ = m.mesh.skin(&bones);
                    }
                }
            }
        }
    }));
    assert!(result.is_ok(), "loading a mutated asset panicked");
}

fn mutate_numbers(v: &mut serde_json::Value, rng: &mut SplitMix64, p: f64) {
    match v {
        serde_json::Value::Number(_) if rng.uniform(0.0, 1.0) < p as f32 => {
            let choices = [
                serde_json::json!(0),
                serde_json::json!(-1),
                serde_json::json!(1),
                serde_json::json!(3),
                serde_json::json!(65535),
                serde_json::json!(1u64 << 31),
                serde_json::json!(u64::MAX),
                serde_json::json!(1e30),
            ];
            *v = choices[rng.below(choices.len())].clone();
        }
        serde_json::Value::Array(a) => a.iter_mut().for_each(|x| mutate_numbers(x, rng, p)),
        serde_json::Value::Object(o) => o.values_mut().for_each(|x| mutate_numbers(x, rng, p)),
        _ => {}
    }
}

#[test]
fn mutated_documents_never_panic() {
    let text = std::fs::read_to_string(data().join("tentacle_external.gltf")).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut rng = SplitMix64::new(0x61F);
    for _ in 0..1500 {
        let mut d = doc.clone();
        mutate_numbers(&mut d, &mut rng, 0.01);
        exercise(d.to_string().as_bytes());
    }
}

#[test]
fn corrupted_binaries_never_panic() {
    let original = std::fs::read(data().join("tentacle.glb")).unwrap();
    let mut rng = SplitMix64::new(0xB1B);
    for _ in 0..1500 {
        let mut bytes = original.clone();
        for _ in 0..1 + rng.below(8) {
            let at = rng.below(bytes.len());
            bytes[at] = rng.next_u64() as u8;
        }
        exercise(&bytes);
    }
}

#[test]
fn hostile_structures_are_rejected() {
    let text = std::fs::read_to_string(data().join("tentacle.gltf")).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let load = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut d = doc.clone();
        edit(&mut d);
        GltfScene::from_slice(d.to_string().as_bytes(), Some(&data()))
    };
    assert!(load(&|_| {}).is_ok());
    // A cycle in the node hierarchy.
    assert!(load(&|d| d["nodes"][2]["children"] = serde_json::json!([0])).is_err());
    // An accessor reading past its buffer view.
    assert!(load(&|d| d["accessors"][0]["count"] = serde_json::json!(100_000)).is_err());
    // A view past the end of its buffer.
    assert!(load(&|d| d["bufferViews"][0]["byteOffset"] = serde_json::json!(1 << 20)).is_err());
    // Buffers outside the asset's folder.
    for uri in ["../../Cargo.toml", "/etc/passwd", "file:///etc/passwd"] {
        assert!(
            load(&|d| d["buffers"][0]["uri"] = serde_json::json!(uri)).is_err(),
            "{uri}"
        );
    }
    // An index past the vertices: valid accessor, wrong contents.
    let s = load(&|d| {
        d["meshes"][0]["primitives"][0]["indices"] = serde_json::json!(0);
        d["accessors"][0]["componentType"] = serde_json::json!(5125);
        d["accessors"][0]["type"] = serde_json::json!("SCALAR");
    });
    assert!(s.is_err() || s.unwrap().mesh(0).is_err());
}
