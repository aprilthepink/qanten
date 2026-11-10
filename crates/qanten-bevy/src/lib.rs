//! # qanten-bevy
//!
//! [Bevy](https://bevy.org) plugins that run `qanten-graphics` workloads on a
//! Q.ANT photonic NPU (or the built-in simulator):
//!
//! | Plugin / item | What it does |
//! |---|---|
//! | [`QantenPlugin`] | opens an NPU ([`Npu::from_env`]) as the [`NpuResource`] and logs the backend |
//! | [`OceanPlugin`], [`Ocean`], [`Buoyancy`] | updates Gerstner ocean meshes every frame on the NPU; floats entities on the waves |
//! | [`NpuSkinnedCollider`] | CPU-side skinned copies of animated meshes for picking and collision, skinned in one NPU call per mesh |
//! | [`ProbeBaker`] | bakes light-probe volumes into spherical harmonics and a 3-D [`Image`](bevy::image::Image) |
//!
//! ```no_run
//! use bevy::prelude::*;
//! use qanten_bevy::{NpuResource, Ocean, OceanPlugin, QantenPlugin};
//! use qanten_graphics::glam::Vec2;
//! use qanten_graphics::ocean::{GerstnerOcean, Grid, wind_sea};
//!
//! fn setup(mut commands: Commands, npu: Res<NpuResource>, mut meshes: ResMut<Assets<Mesh>>) {
//!     let waves = wind_sea(48, Vec2::new(1.0, 0.3), 40.0, 0.7, 1);
//!     let grid = Grid::new(Vec2::splat(-50.0), Vec2::splat(100.0), 128, 128);
//!     let ocean = GerstnerOcean::new(waves, grid).unwrap().with_npu(&npu).unwrap();
//!     let ocean = Ocean::new(ocean, &mut meshes);
//!     commands.spawn((ocean.mesh3d(), ocean));
//! }
//!
//! App::new()
//!     .add_plugins((MinimalPlugins, AssetPlugin::default(), QantenPlugin::default(), OceanPlugin))
//!     .init_asset::<Mesh>()
//!     .add_systems(Startup, setup)
//!     .run();
//! ```
//!
//! `qanten-graphics` uses its own `glam` version; [`to_bevy`] and
//! [`from_bevy`] convert vectors and matrices between the two.

#![warn(missing_docs)]

mod collider;
mod ocean;
mod probes;

use bevy::prelude::*;
use qanten::Npu;

pub use collider::{NpuSkinnedCollider, SkinnedColliderPlugin};
pub use ocean::{Buoyancy, Ocean, OceanPlugin};
pub use probes::{ProbeBaker, ProbeVolume};

/// The NPU every qanten-bevy system uses.
#[derive(Resource, Debug, Clone, Deref)]
pub struct NpuResource(pub Npu);

/// Inserts an [`NpuResource`]: the given NPU, or [`Npu::from_env`]`(0)`
/// (hardware if the driver is available, the simulator otherwise;
/// `QANTEN_BACKEND` overrides). Logs which backend runs.
#[derive(Debug, Clone, Default)]
pub struct QantenPlugin {
    /// Use this NPU instead of opening one from the environment.
    pub npu: Option<Npu>,
}

impl QantenPlugin {
    /// Uses `npu` (for example `Npu::simulator()` in tests).
    pub fn with_npu(npu: &Npu) -> Self {
        Self {
            npu: Some(npu.clone()),
        }
    }
}

impl Plugin for QantenPlugin {
    fn build(&self, app: &mut App) {
        let npu = match &self.npu {
            Some(npu) => npu.clone(),
            None => Npu::from_env(0).unwrap_or_else(|e| {
                warn!("qanten: could not open an NPU ({e}); using the simulator");
                Npu::simulator()
            }),
        };
        info!("qanten: {}", npu.backend().name());
        app.insert_resource(NpuResource(npu));
    }
}

/// Converts between `qanten-graphics`' `glam` types and Bevy's (the two
/// crates may use different `glam` versions).
pub trait Convert<T> {
    /// The converted value.
    fn convert(self) -> T;
}

macro_rules! convert {
    ($($from:ty => $to:ty, $via:ident, $back:ident;)*) => {$(
        impl Convert<$to> for $from {
            fn convert(self) -> $to {
                <$to>::$back(self.$via())
            }
        }
    )*};
}

convert! {
    qanten_graphics::glam::Vec3 => Vec3, to_array, from_array;
    Vec3 => qanten_graphics::glam::Vec3, to_array, from_array;
    qanten_graphics::glam::Vec2 => Vec2, to_array, from_array;
    Vec2 => qanten_graphics::glam::Vec2, to_array, from_array;
}

impl Convert<Mat4> for qanten_graphics::glam::Mat4 {
    fn convert(self) -> Mat4 {
        Mat4::from_cols_array(&self.to_cols_array())
    }
}

impl Convert<qanten_graphics::glam::Mat4> for Mat4 {
    fn convert(self) -> qanten_graphics::glam::Mat4 {
        qanten_graphics::glam::Mat4::from_cols_array(&self.to_cols_array())
    }
}

/// A `qanten-graphics` value as the Bevy `glam` type.
pub fn to_bevy<T, U>(v: T) -> U
where
    T: Convert<U>,
{
    v.convert()
}

/// A Bevy value as the `qanten-graphics` `glam` type.
pub fn from_bevy<T, U>(v: T) -> U
where
    T: Convert<U>,
{
    v.convert()
}
