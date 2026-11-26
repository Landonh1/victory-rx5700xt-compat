//! Checked dispatch sizing and vertex-ID mapping for the software mesh path.
//!
//! Mesh output rows use 16-byte signature registers. The fallback stores the
//! vertex and primitive rows separately, retains the source primitive's
//! three indices, and expands each primitive to three draw vertices. The
//! generated vertex shader can then fetch both per-vertex and per-primitive
//! values without changing the pixel shader's signature.
//!
//! This is the CPU-side ABI foundation only. It does not emit a compute shader
//! or submit a fallback draw; callers must not treat a plan as proof that a
//! shader is translatable.

/// Buffer and draw sizes for one fixed-count software mesh dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshFallbackPlan {
    /// Maximum vertices written by one mesh workgroup.
    pub max_vertices: u32,
    /// Maximum primitives written by one mesh workgroup.
    pub max_primitives: u32,
    /// Bytes occupied by the vertex signature rows for one workgroup.
    pub vertex_bytes_per_group: u64,
    /// Bytes occupied by indices and primitive signature rows for one group.
    pub primitive_bytes_per_group: u64,
    /// Total vertex record storage for the dispatch.
    pub vertex_buffer_bytes: u64,
    /// Total primitive index and signature storage for the dispatch.
    pub primitive_buffer_bytes: u64,
    /// Fixed draw count: three expanded vertices per possible primitive.
    pub draw_vertex_count: u32,
    /// Number of mesh workgroups in the x dimension.
    pub dispatch_groups: u32,
    vertex_stride: u32,
    primitive_stride: u32,
}

/// Location decoded from a conventional draw's `SV_VertexID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandedVertex {
    /// Source mesh workgroup in the x dimension.
    pub group: u32,
    /// Source primitive within that workgroup.
    pub primitive: u32,
    /// Corner within the triangle, from 0 through 2.
    pub corner: u8,
}

impl MeshFallbackPlan {
    /// Derives signature row counts from an already translated mesh shader.
    pub fn from_shader(shader: &super::HostShader, dispatch_groups: u32) -> Result<Self, String> {
        if shader.stage != crate::HwStage::Ms {
            return Err("software mesh layout requires a mesh shader".into());
        }
        let container = crate::dxbc::Container::parse(&shader.dxil)
            .map_err(|e| format!("invalid translated mesh container: {e}"))?;
        let rows = |part| -> Result<u32, String> {
            let Some(part) = container.part(part) else {
                return Ok(0);
            };
            super::signature(part.data)
                .map_err(|e| format!("invalid mesh output signature: {e}"))?
                .iter()
                .filter_map(|e| e.row)
                .try_fold(0u32, |count, row| {
                    row.checked_add(1)
                        .map(|end| count.max(end))
                        .ok_or_else(|| "mesh signature row count overflow".to_owned())
                })
        };
        let vertex_rows = rows(crate::dxbc::FourCC::OSG1)?;
        let primitive_rows = rows(crate::dxbc::FourCC::PSG1)?;
        let psv = container
            .part(crate::dxbc::FourCC::PSV0)
            .ok_or_else(|| "translated mesh shader has no mesh state".to_owned())?;
        if crate::dxbc::u32_at(psv.data, 0).map_err(|e| e.to_string())? < 48 {
            return Err("translated mesh shader has no runtime mesh state".into());
        }
        let max_vertices = u32::from(crate::dxbc::u16_at(psv.data, 16).map_err(|e| e.to_string())?);
        let max_primitives =
            u32::from(crate::dxbc::u16_at(psv.data, 18).map_err(|e| e.to_string())?);
        Self::new(
            max_vertices,
            max_primitives,
            vertex_rows,
            primitive_rows,
            dispatch_groups,
        )
        .map_err(str::to_owned)
    }

    /// Builds the checked buffer layout for one direct dispatch.
    ///
    /// Row counts include any gaps in the signature's register numbering.
    /// A zero row count is valid for a signature with no user rows. Resource
    /// allocation limits remain adapter-specific and are checked by the host.
    pub fn new(
        max_vertices: u32,
        max_primitives: u32,
        vertex_rows: u32,
        primitive_rows: u32,
        dispatch_groups: u32,
    ) -> Result<Self, &'static str> {
        if max_vertices == 0 || max_primitives == 0 {
            return Err("mesh output limits must be nonzero");
        }
        let vertex_stride = vertex_rows
            .checked_mul(16)
            .ok_or("mesh vertex signature stride overflow")?;
        let primitive_stride = primitive_rows
            .checked_mul(16)
            .ok_or("mesh primitive signature stride overflow")?;

        let vertex_bytes_per_group = u64::from(max_vertices)
            .checked_mul(u64::from(vertex_stride))
            .ok_or("mesh vertex buffer size overflow")?;
        // Three u32 indices are retained per primitive. The vertex shader
        // uses them to fetch the source vertex for each expanded corner.
        let index_bytes = u64::from(max_primitives)
            .checked_mul(3 * u64::from(u32::BITS / 8))
            .ok_or("mesh index buffer size overflow")?;
        let primitive_data_bytes = u64::from(max_primitives)
            .checked_mul(u64::from(primitive_stride))
            .ok_or("mesh primitive buffer size overflow")?;
        let primitive_bytes_per_group = index_bytes
            .checked_add(primitive_data_bytes)
            .ok_or("mesh primitive buffer size overflow")?;
        let vertex_buffer_bytes = vertex_bytes_per_group
            .checked_mul(u64::from(dispatch_groups))
            .ok_or("mesh vertex buffer size overflow")?;
        let primitive_buffer_bytes = primitive_bytes_per_group
            .checked_mul(u64::from(dispatch_groups))
            .ok_or("mesh primitive buffer size overflow")?;
        let draw_vertex_count = u64::from(max_primitives)
            .checked_mul(u64::from(dispatch_groups))
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("expanded mesh draw exceeds the D3D12 vertex-count field")?;

        Ok(Self {
            max_vertices,
            max_primitives,
            vertex_bytes_per_group,
            primitive_bytes_per_group,
            vertex_buffer_bytes,
            primitive_buffer_bytes,
            draw_vertex_count,
            dispatch_groups,
            vertex_stride,
            primitive_stride,
        })
    }

    /// Maps a draw vertex ID to the source workgroup, primitive, and corner.
    pub fn expanded_vertex(&self, vertex_id: u32) -> Option<ExpandedVertex> {
        if vertex_id >= self.draw_vertex_count {
            return None;
        }
        let primitive_id = vertex_id / 3;
        Some(ExpandedVertex {
            group: primitive_id / self.max_primitives,
            primitive: primitive_id % self.max_primitives,
            corner: (vertex_id % 3) as u8,
        })
    }

    /// Byte offset of a source vertex record in the dispatch buffer.
    pub fn vertex_offset(&self, group: u32, vertex: u32) -> Option<u64> {
        if group >= self.dispatch_groups || vertex >= self.max_vertices {
            return None;
        }
        u64::from(group)
            .checked_mul(self.vertex_bytes_per_group)?
            .checked_add(u64::from(vertex).checked_mul(u64::from(self.vertex_stride))?)
    }

    /// Byte offset of the three source indices for a primitive.
    pub fn indices_offset(&self, group: u32, primitive: u32) -> Option<u64> {
        if group >= self.dispatch_groups || primitive >= self.max_primitives {
            return None;
        }
        let index_bytes = u64::from(self.max_primitives).checked_mul(12)?;
        let within_group = u64::from(primitive).checked_mul(12)?;
        if within_group.checked_add(12)? > index_bytes {
            return None;
        }
        u64::from(group)
            .checked_mul(self.primitive_bytes_per_group)?
            .checked_add(within_group)
    }

    /// Byte offset of a primitive signature record in the dispatch buffer.
    pub fn primitive_offset(&self, group: u32, primitive: u32) -> Option<u64> {
        if group >= self.dispatch_groups || primitive >= self.max_primitives {
            return None;
        }
        let group_base = u64::from(group).checked_mul(self.primitive_bytes_per_group)?;
        let indices_bytes = u64::from(self.max_primitives).checked_mul(12)?;
        group_base
            .checked_add(indices_bytes)?
            .checked_add(u64::from(primitive).checked_mul(u64::from(self.primitive_stride))?)
    }
}

#[cfg(test)]
mod tests {
    use super::MeshFallbackPlan;
    use crate::HwStage;
    use crate::emit::HostShader;
    use crate::emit::container::{self, Psv, SigElement};

    fn signature_element(row: u32) -> SigElement {
        SigElement {
            name: "ATTRIBUTE".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0xf,
            half: false,
        }
    }

    fn translated_mesh(vertex_row: u32, primitive_row: u32) -> HostShader {
        let outputs = [signature_element(vertex_row)];
        let prims = [signature_element(primitive_row)];
        let psv = container::psv(&Psv {
            stage: 13,
            flag: true,
            inputs: &[],
            outputs: &outputs,
            prims: &prims,
            mesh: Some((
                super::super::shader::Mesh {
                    max_vertices: 64,
                    max_primitives: 32,
                    topology: 2,
                },
                0,
            )),
            gs: None,
            bindings: &[],
            threads: [64, 1, 1],
            wave_size: None,
        });
        let osg = container::signature(&outputs, false);
        let psg = container::signature(&prims, false);
        let dxil = container::build(&[
            (b"OSG1", &osg),
            (b"PSG1", &psg),
            (b"PSV0", &psv),
            (b"DXIL", &[]),
        ]);
        HostShader {
            stage: HwStage::Ms,
            dxil,
            user_data: vec![],
            samplers: vec![],
            static_samplers: vec![],
            root_cbvs: vec![],
            root_srvs: vec![],
            root_uavs: vec![],
            memory: false,
            memory_writes: false,
            view_addresses: false,
            heap: false,
            heap_writes: false,
            view_marks: false,
        }
    }

    #[test]
    fn sizes_per_group_and_expands_primitives_in_order() {
        let plan = MeshFallbackPlan::new(5, 3, 2, 1, 2).unwrap();
        assert_eq!(plan.vertex_bytes_per_group, 160);
        assert_eq!(plan.primitive_bytes_per_group, 3 * 12 + 3 * 16);
        assert_eq!(plan.vertex_buffer_bytes, 320);
        assert_eq!(plan.primitive_buffer_bytes, 168);
        assert_eq!(plan.draw_vertex_count, 18);
        assert_eq!(plan.vertex_offset(1, 4), Some(288));
        assert_eq!(plan.indices_offset(1, 2), Some(108));
        assert_eq!(plan.primitive_offset(1, 2), Some(152));
        assert_eq!(plan.expanded_vertex(8).unwrap().group, 0);
        assert_eq!(plan.expanded_vertex(8).unwrap().primitive, 2);
        assert_eq!(plan.expanded_vertex(8).unwrap().corner, 2);
        assert_eq!(plan.expanded_vertex(9).unwrap().group, 1);
        assert_eq!(plan.expanded_vertex(9).unwrap().primitive, 0);
        assert_eq!(plan.expanded_vertex(9).unwrap().corner, 0);
        assert_eq!(plan.expanded_vertex(18), None);
    }

    #[test]
    fn refuses_bad_strides_and_draw_count_overflow() {
        assert_eq!(
            MeshFallbackPlan::new(4, 2, u32::MAX, 16, 1).unwrap_err(),
            "mesh vertex signature stride overflow"
        );
        assert_eq!(
            MeshFallbackPlan::new(1, 1_000_000_000, 0, 0, 2).unwrap_err(),
            "expanded mesh draw exceeds the D3D12 vertex-count field"
        );
    }

    #[test]
    fn derives_row_aligned_buffers_from_translated_signatures() {
        let plan = MeshFallbackPlan::from_shader(&translated_mesh(2, 1), 4).unwrap();
        assert_eq!(plan.max_vertices, 64);
        assert_eq!(plan.max_primitives, 32);
        assert_eq!(plan.vertex_bytes_per_group, 64 * 48);
        assert_eq!(plan.primitive_bytes_per_group, 32 * 12 + 32 * 32);
        assert_eq!(plan.draw_vertex_count, 32 * 4 * 3);
    }
}
