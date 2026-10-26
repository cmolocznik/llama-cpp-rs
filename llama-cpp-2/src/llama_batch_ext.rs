//! Safe wrapper around `llama_batch_ext`.
//!
//! An extended batch carries rows that are token ids or input embeddings, in any order. A
//! sequence that interleaves text with encoded media is therefore processed by one call, so
//! attention and pooling see the whole sequence. Processing it chunk by chunk instead pools
//! only the last chunk, because pooled outputs are recomputed on every call.

use crate::context::LlamaContext;
use crate::token::LlamaToken;
use llama_cpp_sys_2::{llama_embd, llama_pos, llama_seq_id};
use std::ptr::NonNull;

/// Errors that can occur when building an extended batch.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchExtError {
    /// llama.cpp returned no batch.
    #[error("the extended batch could not be allocated")]
    Allocation,
    /// The batch holds as many rows as the context admits.
    #[error("the extended batch is full")]
    Full,
    /// The token id or the embedding row was refused.
    #[error("the row's token id or embedding was refused")]
    InvalidRow,
    /// The sequence id is outside the context's sequences.
    #[error("the sequence id is outside the context's sequences")]
    InvalidSequence,
    /// An embedding row is not one row of the model's input width.
    #[error("an embedding row has {got} values; the model's input width is {expected}")]
    RowWidth {
        /// The values supplied.
        got: usize,
        /// The model's input embedding width.
        expected: usize,
    },
    /// The model positions an embedding row in several sections (M-RoPE); one position cannot
    /// place it.
    #[error("the model positions embedding rows in {0} sections")]
    SectionedPositions(usize),
    /// The position or the output flag could not be set on the added row.
    #[error("the row's position or output flag was refused")]
    Placement,
}

/// A batch of token and embedding rows for the context it was built from.
#[derive(Debug)]
pub struct LlamaBatchExt {
    batch: NonNull<llama_cpp_sys_2::llama_batch_ext>,
    context: *const llama_cpp_sys_2::llama_context,
    n_embd_inp: usize,
    n_pos_per_embd: usize,
    rows: usize,
}

// SAFETY: The batch owns copies of every row it holds; nothing in it is bound to a thread.
unsafe impl Send for LlamaBatchExt {}

impl LlamaBatchExt {
    /// An empty batch for `context`. It can only be processed by that context.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::Allocation`] if llama.cpp returns no batch.
    pub fn new(context: &LlamaContext<'_>) -> Result<Self, BatchExtError> {
        // SAFETY: `llama_batch_ext_init` reads the context's shape and memory handle; it
        // mutates nothing in the context.
        let raw = unsafe { context.synchronizable_ptr() };
        let batch = NonNull::new(unsafe { llama_cpp_sys_2::llama_batch_ext_init(raw) })
            .ok_or(BatchExtError::Allocation)?;
        let model = unsafe { llama_cpp_sys_2::llama_get_model(raw) };
        let n_embd_inp = unsafe { llama_cpp_sys_2::llama_model_n_embd_inp(model) };
        // llama.cpp reads this many positions for an embedding row (`llama_hparams::n_pos_per_embd`).
        let n_pos_per_embd = match unsafe { llama_cpp_sys_2::llama_model_rope_type(model) } {
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_MROPE | llama_cpp_sys_2::LLAMA_ROPE_TYPE_IMROPE => 4,
            _ => 1,
        };
        Ok(Self {
            batch,
            context: raw.cast_const(),
            n_embd_inp: usize::try_from(n_embd_inp).map_err(|_| BatchExtError::Allocation)?,
            n_pos_per_embd,
            rows: 0,
        })
    }

    /// The width of one embedding row: the model's input embedding width.
    #[must_use]
    pub fn n_embd_inp(&self) -> usize {
        self.n_embd_inp
    }

    /// The rows added since the batch was built or last cleared.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Whether the batch holds no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Adds a token row of `seq` at `pos`, returning its index in the batch.
    ///
    /// # Errors
    ///
    /// The [`BatchExtError`] llama.cpp reported for the row.
    pub fn add_token(
        &mut self,
        token: LlamaToken,
        pos: llama_pos,
        seq: llama_seq_id,
        output: bool,
    ) -> Result<i32, BatchExtError> {
        let idx = unsafe {
            llama_cpp_sys_2::llama_batch_ext_add_token(self.batch.as_ptr(), seq, token.0)
        };
        self.place(idx, pos, output)
    }

    /// Adds one input-embedding row of `seq` at `pos`, returning its index in the batch. The
    /// batch copies the row.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::SectionedPositions`] on a model that positions embedding rows in several
    /// sections; [`BatchExtError::RowWidth`] unless `row` is one row of the model's input width;
    /// else the [`BatchExtError`] llama.cpp reported for the row.
    pub fn add_embedding(
        &mut self,
        row: &[f32],
        pos: llama_pos,
        seq: llama_seq_id,
        output: bool,
    ) -> Result<i32, BatchExtError> {
        if self.n_pos_per_embd != 1 {
            return Err(BatchExtError::SectionedPositions(self.n_pos_per_embd));
        }
        if row.len() != self.n_embd_inp {
            return Err(BatchExtError::RowWidth {
                got: row.len(),
                expected: self.n_embd_inp,
            });
        }
        let embd = llama_embd {
            data: row.as_ptr(),
            n_rows: 1,
            n_embd: row.len(),
        };
        let idx =
            unsafe { llama_cpp_sys_2::llama_batch_ext_add_embd(self.batch.as_ptr(), seq, embd) };
        self.place(idx, pos, output)
    }

    /// Removes every row.
    pub fn clear(&mut self) {
        unsafe { llama_cpp_sys_2::llama_batch_ext_clear(self.batch.as_ptr()) };
        self.rows = 0;
    }

    fn place(&mut self, idx: i32, pos: llama_pos, output: bool) -> Result<i32, BatchExtError> {
        match idx {
            -1 => return Err(BatchExtError::Full),
            -2 => return Err(BatchExtError::InvalidRow),
            -3 => return Err(BatchExtError::InvalidSequence),
            i if i < 0 => return Err(BatchExtError::InvalidRow),
            _ => {}
        }
        // One position: a token row always takes one, and an embedding row reached here only on
        // a model whose embedding rows take one.
        let placed = unsafe {
            llama_cpp_sys_2::llama_batch_ext_set_pos(self.batch.as_ptr(), idx, &raw const pos)
                && llama_cpp_sys_2::llama_batch_ext_set_output_embd(
                    self.batch.as_ptr(),
                    idx,
                    output,
                )
        };
        if !placed {
            return Err(BatchExtError::Placement);
        }
        self.rows += 1;
        Ok(idx)
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut llama_cpp_sys_2::llama_batch_ext {
        self.batch.as_ptr()
    }

    pub(crate) fn context(&self) -> *const llama_cpp_sys_2::llama_context {
        self.context
    }
}

impl Drop for LlamaBatchExt {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_batch_ext_free(self.batch.as_ptr()) }
    }
}
