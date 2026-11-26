# Software mesh-shader fallback

This document records the proposed translation boundary for GPUs that do not
report native D3D12 mesh-shader support. It describes a design, not a supported
runtime path. The fallback must remain opt-in at the pipeline level until each
operation has a faithful lowering and GPU tests cover it.

## Execution paths

Pipeline creation selects one of two paths from the host device's queried
capabilities:

* **Native:** keep the current mesh DXIL, mesh PSO, and `DispatchMesh` path when
  the adapter reports a mesh-shader tier and the required shader model.
* **Compute fallback:** only select this after a mesh shader and its dispatch
  form pass explicit translation and resource checks. A refusal is logged with
  the shader identity and reason; it must not silently drop the mesh dispatch or
  replace missing behavior with constants.

The selected path must be part of the translated-shader cache key. A native
mesh DXIL cache entry cannot be reused as a compute shader, and fallback output
must never be selected for a device that supports the native path.

## Candidate data flow

For a supported triangle mesh shader:

1. Translate the mesh program's ordinary instructions, resource reads, user
   data, and wave behavior with the existing GFX10 front end.
2. Lower mesh output operations to bounded writes into per-dispatch output
   buffers. Preserve the source group ID in the output address so each compute
   group owns a disjoint output range.
3. Order the compute writes before vertex/index reads with D3D12 resource
   transitions or UAV barriers appropriate to the resources and command-list
   sequence.
4. Run a generated vertex shader with a fixed draw count of three vertices per
   possible primitive. `SV_VertexID` maps each draw vertex to its workgroup,
   primitive, and triangle corner. The shader reads the primitive's three
   generated indices, fetches the selected vertex rows, and writes per-primitive
   rows for all three corners with the pixel shader's existing signature and
   flat interpolation. This expansion is required because indexed vertices
   cannot carry distinct per-primitive attributes when shared by triangles.
5. Draw through the existing raster, blend, depth, and render-target state.

The initial row ABI is 16 bytes per signature row, with vertex and primitive
rows stored separately. Each primitive record begins with three 32-bit indices,
followed by its primitive rows. The fallback must preserve the packed export's
index and cull semantics when writing those indices. Draw sizing and all byte
offsets are computed with checked arithmetic; the host must also reject a plan
when adapter resource limits or allocation fail. A CPU-side helper derives
signature row counts from translated mesh DXIL and performs checked sizing and
vertex-ID mapping, but it does not emit shaders or select a runtime path.

## Amplification and indirect work

Amplification shaders are not represented by the current accepted pipeline
stages. They require a separate translation path. A possible later design is a
compute pass that writes a work list and payload records, followed by a
mesh-compute pass that consumes the list. That requires explicit handling for
child counts, payload lifetime, group ordering, indirect arguments, and the
barriers between passes. Until those rules are implemented and tested, a
pipeline containing an amplification stage is unsupported by the fallback.

Indirect mesh dispatch also needs its own design. Reinterpreting a
`DispatchMesh` argument record as `Dispatch` arguments is insufficient: the
fallback must size output storage and produce a matching indirect draw count
without reading GPU-generated counts on the CPU. GTA VI's world path uses
indirect work, so a direct-dispatch-only prototype would not establish game
compatibility.

## Initial support boundary

The first implementation should be a diagnostic census, then a compute output
ABI tested with synthetic triangle shaders. Do not route live title pipelines
to it until these cases are verified:

* triangle topology and output counts within checked buffer limits;
* vertex and primitive output rows, including packed half values;
* all three indices and primitive culling;
* descriptor heaps, root views, title-memory accesses, and resource state
  transitions;
* wave operations and LDS/barrier behavior used by the shader;
* direct and indirect dispatch sizing;
* generated PSO creation and image equivalence against native mesh output on a
  mesh-capable reference device.

The current initial fallback does not claim support for amplification shaders,
non-triangle topology, mesh shaders whose behavior depends on unsupported wave
semantics, or indirect dispatch until their translation and submission paths
are implemented. A census result is a count of discovered or natively
translatable programs; it is not a fallback compatibility count.
