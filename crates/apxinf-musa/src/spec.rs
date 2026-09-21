//! The identity of a call, and the policy that constrains which candidate may serve it.
//!
//! Kept separate on purpose. `Spec` is what makes two calls the *same* operation
//! for candidate-selection purposes -- so it must contain everything that can
//! change the arithmetic and nothing that cannot. `Policy` is what the caller
//! requires regardless of speed. Mixing the two is how a tuning cache ends up
//! keyed on a calibration scale value, which is why the CUDA layer keeps
//! structural predicates here and scale *values* in the bindings
//! (`crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h:37`).

/// The data type a stage is computed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    Bf16,
    F8E4M3,
    I8,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F8E4M3 => "f8e4m3",
            Self::I8 => "i8",
        }
    }
}

/// How a quantized operand is scaled.
///
/// Structural only: a row scale and a channel scale are different *shapes*, so
/// they select different implementations. The numbers themselves live in the
/// bindings, not here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleKind {
    None,
    Unit,
    Row,
    Channel,
    RowChannel,
}

impl ScaleKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Unit => "unit",
            Self::Row => "row",
            Self::Channel => "channel",
            Self::RowChannel => "row_channel",
        }
    }
}

/// Memory order of the operands as the caller presents them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Contiguous row-major, the public contract of every ApxInf operator.
    RowMajor,
    /// One of the transposed or packed layouts an engine may prepare.
    Prepared,
}

/// Everything that can change which candidate is admissible, and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    /// Rows of the activation.
    pub rows: usize,
    /// Output columns.
    pub columns: usize,
    /// Contracted dimension.
    pub inner: usize,
    pub input: DType,
    pub output: DType,
    /// Precision the contraction accumulates in.
    pub accumulation: DType,
    pub input_scale: ScaleKind,
    pub weight_scale: ScaleKind,
    pub layout: Layout,
    /// Number of independent attention windows in the sequence, for the
    /// operations that carry one.
    pub windows: usize,
}

impl Spec {
    /// A plain dense operation over `[rows, inner] x [inner, columns]`.
    pub fn dense(rows: usize, inner: usize, columns: usize, dtype: DType) -> Self {
        Self {
            rows,
            columns,
            inner,
            input: dtype,
            output: dtype,
            accumulation: DType::F32,
            input_scale: ScaleKind::None,
            weight_scale: ScaleKind::None,
            layout: Layout::RowMajor,
            windows: 1,
        }
    }

    /// Note the shape of a candidate-selection key: no scale *values*, no
    /// pointer, no stream, nothing that varies between two calls whose
    /// arithmetic is identical.
    pub fn key(&self) -> String {
        format!(
            "{}x{}x{}:{}/{}/{}/{}/{}/{}:w{}",
            self.rows,
            self.columns,
            self.inner,
            self.input.name(),
            self.output.name(),
            self.accumulation.name(),
            self.input_scale.name(),
            self.weight_scale.name(),
            match self.layout {
                Layout::RowMajor => "row",
                Layout::Prepared => "prepared",
            },
            self.windows,
        )
    }
}

/// What the caller requires of any candidate, independent of how fast it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    /// The candidate's output must be identical between two runs with the same
    /// inputs. Attention reductions and atomics are where this stops being free.
    pub deterministic: bool,
    /// The candidate must be safe to run inside a captured graph: no allocation,
    /// no synchronisation, no host callback.
    pub graph_safe: bool,
    /// Upper bound on workspace the candidate may request.
    pub workspace_limit: usize,
    /// Whether a candidate that cannot be reached should be reported as a gap
    /// rather than silently skipped. Off by default is what makes silent gaps
    /// possible, so it is on by default here.
    pub require_candidate: bool,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            deterministic: true,
            graph_safe: true,
            workspace_limit: usize::MAX,
            require_candidate: true,
        }
    }
}
