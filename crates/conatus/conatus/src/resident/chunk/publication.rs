// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl<I> ResidentChunk<I> {
    pub fn new(
        client: ResidentClient,
        identity: I,
        bounds: ChunkBounds,
        revision: u64,
        valid_read_epoch: ReadEpoch,
        dirty_regions: Vec<DirtyRegion>,
    ) -> Self {
        Self {
            client,
            identity,
            bounds,
            stamp: ChunkStamp {
                revision,
                valid_read_epoch,
            },
            dirty_regions,
            planes: BTreeMap::new(),
        }
    }

    pub fn identity(&self) -> &I {
        &self.identity
    }

    pub const fn bounds(&self) -> ChunkBounds {
        self.bounds
    }

    pub const fn stamp(&self) -> ChunkStamp {
        self.stamp
    }

    pub fn dirty_regions(&self) -> &[DirtyRegion] {
        &self.dirty_regions
    }

    pub fn plane_ids(&self) -> impl ExactSizeIterator<Item = &PlaneId> {
        self.planes.keys()
    }

    /// Allocate and initialize one contiguous 3D plane through CubeCL.
    pub fn insert_plane<T: ResidentElement>(
        &mut self,
        id: PlaneId,
        class: PlaneClass,
        shape: [usize; 3],
        values: &[T],
    ) -> Result<(), ResidentChunkError> {
        if self.planes.contains_key(&id) {
            return Err(ResidentChunkError::DuplicatePlane(id));
        }
        let layout = PlaneLayout::contiguous::<T>(shape)?;
        let expected = layout.element_count();
        if values.len() != expected {
            return Err(ResidentChunkError::ElementCount {
                plane: id,
                expected,
                actual: values.len(),
            });
        }

        let handle = self
            .compute()
            .create_from_slice(bytemuck::cast_slice(values));
        self.planes.insert(
            id,
            ResidentPlane {
                handle,
                layout,
                class,
            },
        );
        Ok(())
    }

    /// Apply one authority-approved patch without replacing the CubeCL allocation.
    ///
    /// The caller owns the record and the shared-device schedule. Conatus checks that
    /// the patch was derived from the resident stamp still being replaced, writes
    /// only the supplied aligned range, then publishes the committed stamp and dirty
    /// regions together. Views minted before this call retain their old stamp and
    /// must be rejected by revision-aware consumers.
    ///
    /// This narrow form accepts a single-plane chunk. A bundle-wide stamp cannot
    /// truthfully advance after one of several planes changes; that case needs one
    /// validated batch commit covering every changed plane.
    #[allow(
        clippy::too_many_arguments,
        reason = "the queue, plane, two stamps, byte range, and dirty publication are distinct validation inputs"
    )]
    pub fn commit_plane_patch<T: ResidentElement>(
        &mut self,
        queue: &wgpu::Queue,
        id: &PlaneId,
        expected_stamp: ChunkStamp,
        element_offset: usize,
        values: &[T],
        committed_stamp: ChunkStamp,
        dirty_regions: Vec<DirtyRegion>,
    ) -> Result<(), ResidentChunkError> {
        self.commit_plane_patches(
            queue,
            id,
            expected_stamp,
            &[PlanePatch::new(element_offset, values)],
            committed_stamp,
            dirty_regions,
        )
    }

    /// Apply several authority-approved ranges as one published resident revision.
    ///
    /// Every range, overlap, stamp, and dirty region is checked before the first
    /// queue write. A validation failure therefore leaves both the allocation and
    /// its stamp unchanged. The queue observes the accepted writes in patch order,
    /// then readers may use `committed_stamp` after the host's submission boundary.
    ///
    /// Like [`Self::commit_plane_patch`], this narrow operation accepts a
    /// single-plane chunk. Updating several planes truthfully requires a separate
    /// bundle-wide commit primitive.
    #[allow(
        clippy::too_many_arguments,
        reason = "the queue, plane, two stamps, sparse ranges, and dirty publication are distinct validation inputs"
    )]
    pub fn commit_plane_patches<T: ResidentElement>(
        &mut self,
        queue: &wgpu::Queue,
        id: &PlaneId,
        expected_stamp: ChunkStamp,
        patches: &[PlanePatch<'_, T>],
        committed_stamp: ChunkStamp,
        dirty_regions: Vec<DirtyRegion>,
    ) -> Result<(), ResidentChunkError> {
        if self.stamp != expected_stamp {
            return Err(ResidentChunkError::StaleStamp {
                expected: expected_stamp,
                actual: self.stamp,
            });
        }
        if committed_stamp.revision <= expected_stamp.revision
            || committed_stamp.valid_read_epoch <= expected_stamp.valid_read_epoch
        {
            return Err(ResidentChunkError::NonAdvancingStamp {
                previous: expected_stamp,
                committed: committed_stamp,
            });
        }
        for region in &dirty_regions {
            let fits = !region.extent.contains(&0)
                && (0..3).all(|axis| {
                    region.origin[axis]
                        .checked_add(region.extent[axis])
                        .is_some_and(|end| end <= self.bounds.extent[axis])
                });
            if !fits {
                return Err(ResidentChunkError::DirtyRegionOutOfBounds {
                    region: *region,
                    chunk_extent: self.bounds.extent,
                });
            }
        }

        if patches.is_empty() {
            return Err(ResidentChunkError::EmptyPatchBatch { plane: id.clone() });
        }

        let plane = self.plane(id)?;
        if self.planes.len() != 1 {
            return Err(ResidentChunkError::PatchRequiresSinglePlane {
                plane_count: self.planes.len(),
            });
        }
        if plane.layout.element_type != T::ELEMENT_TYPE {
            return Err(ResidentChunkError::PatchElementType {
                plane: id.clone(),
                expected: plane.layout.element_type,
                actual: T::ELEMENT_TYPE,
            });
        }

        let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        let mut ranges = Vec::with_capacity(patches.len());
        for (index, patch) in patches.iter().enumerate() {
            let byte_offset = patch
                .element_offset
                .checked_mul(T::ELEMENT_TYPE.byte_width())
                .ok_or_else(|| ResidentChunkError::PatchRangeOverflow { plane: id.clone() })?;
            let byte_len = size_of_val(patch.values);
            let byte_end = byte_offset
                .checked_add(byte_len)
                .ok_or_else(|| ResidentChunkError::PatchRangeOverflow { plane: id.clone() })?;
            if byte_len == 0 || byte_end > plane.layout.byte_len() {
                return Err(ResidentChunkError::PatchRange {
                    plane: id.clone(),
                    byte_offset,
                    byte_len,
                    plane_byte_len: plane.layout.byte_len(),
                });
            }
            if !byte_offset.is_multiple_of(alignment) || !byte_len.is_multiple_of(alignment) {
                return Err(ResidentChunkError::UnalignedPatch {
                    plane: id.clone(),
                    byte_offset,
                    byte_len,
                    required_alignment: alignment,
                });
            }
            ranges.push((byte_offset, byte_end, index));
        }
        ranges.sort_unstable_by_key(|range| (range.0, range.1));
        for pair in ranges.windows(2) {
            let first = pair[0];
            let second = pair[1];
            if first.1 > second.0 {
                return Err(ResidentChunkError::OverlappingPatches {
                    plane: id.clone(),
                    first_index: first.2,
                    second_index: second.2,
                });
            }
        }

        let allocation = self.allocation(&plane.handle)?;
        for patch in patches {
            let byte_offset = patch.element_offset * T::ELEMENT_TYPE.byte_width();
            queue.write_buffer(
                allocation.buffer(),
                allocation.offset() + byte_offset as u64,
                bytemuck::cast_slice(patch.values),
            );
        }
        self.stamp = committed_stamp;
        self.dirty_regions = dirty_regions;
        Ok(())
    }

    /// Construct a raw storage-buffer view without copying the plane.
    pub fn raw_kernel_view(&self, id: &PlaneId) -> Result<RawKernelView, ResidentChunkError> {
        let plane = self.plane(id)?;
        let allocation = self.allocation(&plane.handle)?;
        Ok(RawKernelView {
            allocation,
            layout: plane.layout,
            class: plane.class,
            stamp: self.stamp,
            _lease: plane.handle.clone(),
        })
    }

    /// Construct a Burn `f32` tensor over the same CubeCL handle.
    #[cfg(feature = "burn")]
    ///
    /// This constructor only accepts an `F32` plane. Exact integer planes stay
    /// exact rather than acquiring a lossy float interpretation for convenience.
    pub fn burn_f32_view(&self, id: &PlaneId) -> Result<BurnTensorView, ResidentChunkError> {
        let plane = self.plane(id)?;
        if plane.layout.element_type != PlaneElementType::F32 {
            return Err(ResidentChunkError::BurnElementType {
                plane: id.clone(),
                actual: plane.layout.element_type,
            });
        }
        let allocation = self.allocation(&plane.handle)?;
        let primitive = CubeTensor::new_contiguous(
            self.compute().clone(),
            self.client.device.clone().into(),
            Shape::new(plane.layout.shape),
            plane.handle.clone(),
            DType::F32,
        );
        let tensor = Tensor::from_primitive::<Wgpu>(primitive);
        Ok(BurnTensorView {
            tensor,
            allocation,
            layout: plane.layout,
            class: plane.class,
            stamp: self.stamp,
        })
    }

    pub(super) fn compute(&self) -> &Client {
        &self.client.compute
    }

    pub(super) fn plane(&self, id: &PlaneId) -> Result<&ResidentPlane, ResidentChunkError> {
        self.planes
            .get(id)
            .ok_or_else(|| ResidentChunkError::UnknownPlane(id.clone()))
    }

    pub(super) fn allocation(&self, handle: &Handle) -> Result<BufferIdentity, ResidentChunkError> {
        let managed = self
            .compute()
            .get_resource::<cubecl::wgpu::WgpuServer<cubecl::wgpu::AutoCompiler>>(handle.clone())
            .map_err(|error| ResidentChunkError::Resource(error.to_string()))?;
        let resource = managed.resource();
        Ok(BufferIdentity {
            buffer: resource.buffer.clone(),
            offset: resource.offset,
            size: resource.size,
        })
    }
}
