// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Shared Conatus resident chunk planes and their zero-copy consumer views.
//!
//! Allocation has one direction here: [`ResidentClient`] allocates each
//! plane through CubeCL, then [`ResidentChunk`] lends Burn and raw wgpu
//! arrangements of that allocation. Neither view imports or copies the
//! plane, and this module deliberately offers no CPU whole-plane read.

use std::{
    collections::BTreeMap,
    fmt,
    future::Future,
    mem::size_of_val,
    num::NonZeroU64,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Waker},
};

#[cfg(feature = "burn")]
use burn::tensor::{DType, Shape, Tensor};
#[cfg(feature = "burn")]
use burn_wgpu::{CubeTensor, Wgpu};
use bytemuck::Pod;
use cubecl::wgpu::{RuntimeOptions, WgpuDevice, WgpuSetup, init_device};
use cubecl::{client::Client, server::Handle};

/// Burn's view of one resident `f32` channel plane.
#[cfg(feature = "burn")]
pub type ResidentTensor = Tensor<3>;

/// The host schedule epoch in which a materialized chunk is safe to read.
///
/// This is intentionally a host-issued number, not a frame counter hidden
/// inside Conatus. The shared device scheduler decides when a submitted write
/// becomes visible to its reader tenants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReadEpoch(u64);

impl ReadEpoch {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Integer world bounds for a resident voxel chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkBounds {
    pub origin: [i64; 3],
    pub extent: [u32; 3],
}

/// A chunk-local dirty box, in cell coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRegion {
    pub origin: [u32; 3],
    pub extent: [u32; 3],
}

/// One aligned replacement range inside a resident plane.
///
/// Several patches can be validated and published under one [`ChunkStamp`]
/// with [`ResidentChunk::commit_plane_patches`]. The values are borrowed only
/// for the duration of that queue write.
#[derive(Clone, Copy, Debug)]
pub struct PlanePatch<'a, T> {
    pub element_offset: usize,
    pub values: &'a [T],
}

impl<'a, T> PlanePatch<'a, T> {
    pub const fn new(element_offset: usize, values: &'a [T]) -> Self {
        Self {
            element_offset,
            values,
        }
    }
}

/// The fact-bearing class of a channel plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneClass {
    /// Canonical values such as occupancy, palette id, provenance, or flags.
    Exact,
    /// Replay-exact numerical state stored as fixed point.
    FixedPoint,
    /// Learned or derived state whose drift does not alter the record.
    Derived,
    /// A work plane, candidate result, mask, or queue.
    Temporary,
}

/// The scalar representation held by one channel plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneElementType {
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    F32,
}

impl PlaneElementType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
        }
    }
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
}

/// A scalar that can be uploaded as a resident channel plane.
pub trait ResidentElement: Pod + sealed::Sealed {
    const ELEMENT_TYPE: PlaneElementType;
}

macro_rules! resident_elements {
    ($($ty:ty => $element:ident),+ $(,)?) => {
        $(impl ResidentElement for $ty {
            const ELEMENT_TYPE: PlaneElementType = PlaneElementType::$element;
        })+
    };
}

resident_elements! {
    u8 => U8,
    u16 => U16,
    u32 => U32,
    i8 => I8,
    i16 => I16,
    i32 => I32,
    f32 => F32,
}

/// A named plane in a resident bundle.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlaneId(String);

impl PlaneId {
    pub fn new(name: impl Into<String>) -> Result<Self, ResidentChunkError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ResidentChunkError::EmptyPlaneId);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PlaneId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Shape and scalar layout of one contiguous 3D plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    pub shape: [usize; 3],
    pub strides: [usize; 3],
    pub element_type: PlaneElementType,
}

impl PlaneLayout {
    fn contiguous<T: ResidentElement>(shape: [usize; 3]) -> Result<Self, ResidentChunkError> {
        if shape.contains(&0) {
            return Err(ResidentChunkError::EmptyShape { shape });
        }
        let yz = shape[1]
            .checked_mul(shape[2])
            .ok_or(ResidentChunkError::ShapeOverflow { shape })?;
        shape[0]
            .checked_mul(yz)
            .and_then(|elements| elements.checked_mul(T::ELEMENT_TYPE.byte_width()))
            .ok_or(ResidentChunkError::ShapeOverflow { shape })?;
        Ok(Self {
            shape,
            strides: [yz, shape[2], 1],
            element_type: T::ELEMENT_TYPE,
        })
    }

    pub fn element_count(self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    pub fn byte_len(self) -> usize {
        self.element_count() * self.element_type.byte_width()
    }
}

/// Authoritative snapshot metadata copied into every consumer view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkStamp {
    pub revision: u64,
    pub valid_read_epoch: ReadEpoch,
}

/// Identity of a byte range in one concrete wgpu buffer.
///
/// CubeCL may suballocate several planes from one buffer. Equality therefore
/// checks the wgpu buffer object and the allocation's offset and size, rather
/// than treating any two ranges in the same pooled buffer as one allocation.
#[derive(Clone, Debug)]
pub struct BufferIdentity {
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
}

impl BufferIdentity {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub const fn offset(&self) -> u64 {
        self.offset
    }

    pub const fn size(&self) -> u64 {
        self.size
    }

    pub fn same_buffer(&self, other: &Self) -> bool {
        self.buffer == other.buffer
    }
}

impl PartialEq for BufferIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer && self.offset == other.offset && self.size == other.size
    }
}

impl Eq for BufferIdentity {}

/// One raw-kernel arrangement of a resident plane.
///
/// The private CubeCL handle keeps this allocation range leased for as long
/// as the raw view exists, even if its parent chunk is dropped.
#[derive(Debug)]
pub struct RawKernelView {
    allocation: BufferIdentity,
    layout: PlaneLayout,
    class: PlaneClass,
    stamp: ChunkStamp,
    _lease: Handle,
}

impl RawKernelView {
    pub fn allocation(&self) -> &BufferIdentity {
        &self.allocation
    }

    pub const fn layout(&self) -> PlaneLayout {
        self.layout
    }

    pub const fn class(&self) -> PlaneClass {
        self.class
    }

    pub const fn stamp(&self) -> ChunkStamp {
        self.stamp
    }

    /// Everything a consumer outside this crate needs to read the plane,
    /// and nothing it does not.
    ///
    /// Consumers that must not depend on a compute stack (a renderer, a
    /// tracer) spell the lease in their own plain-wgpu terms; this is the
    /// producing side's single definition of what those terms mean, so a
    /// host assembling one cannot invent a field or drop the stamp. The
    /// byte length is derived here rather than restated, which is the
    /// error the consumer-side spellings cannot make on their own.
    pub fn lease(&self) -> SpatialLease<'_> {
        SpatialLease {
            buffer: self.allocation.buffer(),
            offset: self.allocation.offset(),
            size: self.allocation.size(),
            shape: self.layout.shape,
            element_type: self.layout.element_type,
            stamp: self.stamp,
        }
    }

    /// A storage-buffer binding for the exact CubeCL allocation range.
    pub fn binding(&self) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding {
            buffer: self.allocation.buffer(),
            offset: self.allocation.offset(),
            size: NonZeroU64::new(self.allocation.size().next_multiple_of(4)),
        }
    }
}

/// The producing side's definition of a resident plane handed outward.
///
/// Deliberately plain: a buffer range, the shape and element type that
/// range holds, and the stamp it was materialized at. A consumer needs
/// exactly this much to copy or bind the plane, and needs no part of
/// CubeCL or Burn to understand it.
///
/// `byte_len` is the load-bearing method. Consumers size their copies
/// from `shape`, and a shape that disagrees with the allocation would
/// walk off the end of the lease into whatever the pool put next to it,
/// which is silent corruption rather than a fault. Checking it is the
/// consumer's obligation and this makes it one line.
#[derive(Clone, Copy, Debug)]
pub struct SpatialLease<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
    pub shape: [usize; 3],
    pub element_type: PlaneElementType,
    pub stamp: ChunkStamp,
}

impl SpatialLease<'_> {
    /// The bytes `shape` and `element_type` actually describe.
    pub fn byte_len(&self) -> u64 {
        (self.shape[0] * self.shape[1] * self.shape[2] * self.element_type.byte_width()) as u64
    }

    /// Whether the described shape fits the leased range. A consumer that
    /// copies without asking this can read a neighbouring allocation.
    pub fn fits(&self) -> bool {
        self.byte_len() <= self.size
    }
}

/// One Burn arrangement of a resident `f32` plane.
#[cfg(feature = "burn")]
#[derive(Clone, Debug)]
pub struct BurnTensorView {
    tensor: ResidentTensor,
    allocation: BufferIdentity,
    layout: PlaneLayout,
    class: PlaneClass,
    stamp: ChunkStamp,
}

#[cfg(feature = "burn")]
impl BurnTensorView {
    pub fn tensor(&self) -> &ResidentTensor {
        &self.tensor
    }

    pub fn into_tensor(self) -> ResidentTensor {
        self.tensor
    }

    pub fn allocation(&self) -> &BufferIdentity {
        &self.allocation
    }

    pub const fn layout(&self) -> PlaneLayout {
        self.layout
    }

    pub const fn class(&self) -> PlaneClass {
        self.class
    }

    pub const fn stamp(&self) -> ChunkStamp {
        self.stamp
    }
}

/// CubeCL registered on a host-owned wgpu device and queue.
///
/// [`Self::init`] consumes clones in [`WgpuSetup`]; it does not create a
/// second wgpu device. A host that already registered CubeCL can use
/// [`Self::from_registered_device`] instead.
#[derive(Clone)]
pub struct ResidentClient {
    compute: Client,
    device: WgpuDevice,
    /// The host's wgpu device, when this client was built from it, so a
    /// reader can poll map callbacks without blocking (see
    /// [`Self::poll_device`]).
    wgpu_device: Option<wgpu::Device>,
    /// Readbacks whose reader went away before they finished, polled to the
    /// end here (see [`Self::adopt`]).
    orphans: Arc<Mutex<Vec<Orphan>>>,
}

/// A readback nobody waits for any more, kept alive until it completes.
pub type Orphan = Pin<Box<dyn Future<Output = ()> + Send>>;

impl fmt::Debug for ResidentClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResidentClient")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct ResidentPlane {
    handle: Handle,
    layout: PlaneLayout,
    class: PlaneClass,
}

/// A bundle of typed, GPU-resident channel planes for one world chunk.
///
/// `I` is the world's own chunk identity type. Conatus does not mint a parallel
/// identity model; the wing can carry its existing region key here directly.
#[derive(Debug)]
pub struct ResidentChunk<I> {
    client: ResidentClient,
    identity: I,
    bounds: ChunkBounds,
    stamp: ChunkStamp,
    dirty_regions: Vec<DirtyRegion>,
    planes: BTreeMap<PlaneId, ResidentPlane>,
}

/// Construction and view errors for [`ResidentChunk`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentChunkError {
    EmptyPlaneId,
    EmptyShape {
        shape: [usize; 3],
    },
    ShapeOverflow {
        shape: [usize; 3],
    },
    DuplicatePlane(PlaneId),
    UnknownPlane(PlaneId),
    ElementCount {
        plane: PlaneId,
        expected: usize,
        actual: usize,
    },
    BurnElementType {
        plane: PlaneId,
        actual: PlaneElementType,
    },
    StaleStamp {
        expected: ChunkStamp,
        actual: ChunkStamp,
    },
    NonAdvancingStamp {
        previous: ChunkStamp,
        committed: ChunkStamp,
    },
    DirtyRegionOutOfBounds {
        region: DirtyRegion,
        chunk_extent: [u32; 3],
    },
    PatchRequiresSinglePlane {
        plane_count: usize,
    },
    EmptyPatchBatch {
        plane: PlaneId,
    },
    OverlappingPatches {
        plane: PlaneId,
        first_index: usize,
        second_index: usize,
    },
    PatchElementType {
        plane: PlaneId,
        expected: PlaneElementType,
        actual: PlaneElementType,
    },
    PatchRangeOverflow {
        plane: PlaneId,
    },
    PatchRange {
        plane: PlaneId,
        byte_offset: usize,
        byte_len: usize,
        plane_byte_len: usize,
    },
    UnalignedPatch {
        plane: PlaneId,
        byte_offset: usize,
        byte_len: usize,
        required_alignment: usize,
    },
    Resource(String),
}

#[path = "chunk/publication.rs"]
mod publication;

#[path = "chunk/client.rs"]
mod client;

#[path = "chunk/errors.rs"]
mod errors;
